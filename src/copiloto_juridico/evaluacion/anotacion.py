"""Redacción de consultas de evaluación con el LLM y sus comprobaciones.

- `redactar_borradores`: el LLM propone una consulta al estilo de un defensor para cada entrada nueva. Cada
  borrador pasa por el verificador de fuga (7 palabras seguidas iguales a la sentencia) y se regenera si copia.
- `_problemas`: las comprobaciones de la guía (fuga, nombre del tipo, datos identificables); `referencias` las usa
  para validar las consultas.
"""

from __future__ import annotations

import re
from typing import Callable

import pandas as pd

from .. import prompts
from ..clasificador import etiqueta_tipo
from .conjuntos import Consulta, detalles_identificables, ngramas_compartidos, nombre_tipo_en_consulta

_RE_RESOLUTIVOS = re.compile(r"r\s?e\s?s\s?u\s?e\s?l\s?v\s?e|p\s?r\s?o\s?p\s?o\s?s\s?i\s?c\s?i\s?o\s?n\s?e\s?s|resolutivos")


def _limpiar(texto: str) -> str:
    return texto.strip().strip('"').strip()


_GENERICAS = {"delito", "delitos", "otros", "otro", "contra", "para", "como", "entre", "sobre", "modalidad", "grado",
              "persona", "personas", "comision"}


def palabras_prohibidas(tipo: str | None) -> list[str]:
    """Palabras distintivas del nombre del tipo en el portal ("robo calificado" -> robo, calificado)."""
    nombre = (tipo or "").split("/", 1)[-1]
    return [w for w in dict.fromkeys(re.split(r"[-_]", nombre)) if len(w) >= 4 and w not in _GENERICAS]


def _problemas(c: Consulta, texto: str, prohibidas: list[str]) -> str:
    """Por qué un borrador no sirve (vacío si sirve): no es una consulta, copia frases, nombra el tipo o trae datos
    identificables."""
    from ..pipeline import es_fallo
    from .conjuntos import _palabras

    if es_fallo(c.consulta):
        return "no era una consulta (el modelo se negó o respondió otra cosa)"
    fuga = ngramas_compartidos(c.consulta, texto)
    usadas = sorted(set(_palabras(c.consulta)) & set(prohibidas))
    detalles = detalles_identificables(c.consulta)
    if fuga:
        return f"copiaba estas frases de la sentencia: «{'», «'.join(fuga[:3])}»"
    if nombre_tipo_en_consulta(c) or usadas:
        return f"usaba palabras del nombre del delito ({', '.join(usadas) or etiqueta_tipo(c.tipo)})"
    if detalles:
        return f"incluía datos que identifican la sentencia: {', '.join(detalles[:3])}"
    return ""


def redactar_borradores(plantilla: list[Consulta], textos: pd.Series, llm, n: int | None = None,
                        intentos: int = 3, estricto: bool = False, log: Callable[[str], None] = print) -> list[Consulta]:
    """Borrador de consulta para las primeras `n` entradas de la plantilla (todas si n es None). `estricto`
   : prompt con las palabras del tipo prohibidas y sin números, también verificado."""
    salida = []
    for c in plantilla[:n]:
        c.borrador, c.revisado = True, False
        c.notas = c.notas.split("reescríbelo. ", 1)[-1]  # sin la marca de una corrida anterior
        c.extra.pop("fuga", None)
        if c.sin_respuesta:
            c.consulta = _limpiar(llm.generate(prompts.PROMPT_BORRADOR_SIN_RESPUESTA.format(
                tema=c.extra.get("tema_sugerido", ""))))
        else:
            texto = textos[c.doc_id_fuente]
            prohibidas = palabras_prohibidas(c.tipo) if estricto else []
            base = (prompts.PROMPT_BORRADOR_CONSULTA_ESTRICTO.format(extracto=texto[:6000],
                                                                    prohibidas=", ".join(prohibidas) or "ninguna")
                    if estricto else prompts.PROMPT_BORRADOR_CONSULTA.format(extracto=texto[:6000]))
            prompt, problema = base, ""
            for intento in range(1, intentos + 1):
                c.consulta = _limpiar(llm.generate(prompt))
                problema = _problemas(c, texto, prohibidas)
                if not problema:
                    break
                prompt = base + f"\n(Tu borrador anterior {problema}. Escríbelo de nuevo, con tus palabras.)\nConsulta:"
            c.extra["intentos"] = intento
            if problema:
                c.extra["fuga"] = problema
                c.notas = (f"ATENCIÓN: el borrador {problema}; reescríbelo. " + c.notas)
        log(f"   {c.id}: {c.consulta[:100]}{'  ⚠️' if c.extra.get('fuga') else ''}")
        salida.append(c)
    return salida


