"""Material para las personas que construyen el conjunto de evaluación (no son resultados).

- `redactar_borradores`: el LLM propone una consulta al estilo de un defensor para cada entrada de la plantilla.
  Cada borrador pasa por el verificador de fuga (7 palabras seguidas iguales a la sentencia) y se regenera si
  copia. Todos quedan `borrador: true, revisado: false`: la evaluación en prueba se rechaza hasta revisarlos.
- `hoja_juicios` y `hoja_fichas`: hojas CSV listas para el abogado. El código las lee directamente al llenarse.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from .. import config, prompts
from ..clasificador import etiqueta_tipo
from .conjuntos import Consulta, detalles_identificables, ngramas_compartidos, nombre_tipo_en_consulta

_RE_RESOLUTIVOS = re.compile(r"r\s?e\s?s\s?u\s?e\s?l\s?v\s?e|p\s?r\s?o\s?p\s?o\s?s\s?i\s?c\s?i\s?o\s?n\s?e\s?s|resolutivos")


def _limpiar(texto: str) -> str:
    return texto.strip().strip('"').strip()


def redactar_borradores(plantilla: list[Consulta], textos: pd.Series, llm, n: int | None = None,
                        intentos: int = 3, log: Callable[[str], None] = print) -> list[Consulta]:
    """Borrador de consulta para las primeras `n` entradas de la plantilla (todas si n es None)."""
    salida = []
    for c in plantilla[:n]:
        c.borrador, c.revisado = True, False
        if c.sin_respuesta:
            c.consulta = _limpiar(llm.generate(prompts.PROMPT_BORRADOR_SIN_RESPUESTA.format(
                tema=c.extra.get("tema_sugerido", ""))))
        else:
            texto = textos[c.doc_id_fuente]
            prompt = prompts.PROMPT_BORRADOR_CONSULTA.format(extracto=texto[:6000])
            for intento in range(1, intentos + 1):
                c.consulta = _limpiar(llm.generate(prompt))
                fuga = ngramas_compartidos(c.consulta, texto)
                nombra = nombre_tipo_en_consulta(c)
                detalles = detalles_identificables(c.consulta)
                if not fuga and not nombra and not detalles:
                    break
                problema = (f"copiaba estas frases de la sentencia: «{'», «'.join(fuga[:3])}»" if fuga
                            else f"usaba el nombre del delito '{etiqueta_tipo(c.tipo)}'" if nombra
                            else f"incluía datos que identifican la sentencia: {', '.join(detalles[:3])}")
                prompt = (prompts.PROMPT_BORRADOR_CONSULTA.format(extracto=texto[:6000])
                          + f"\n(Tu borrador anterior {problema}. Escríbelo de nuevo, con tus palabras.)\nConsulta:")
            c.extra["intentos"] = intento
            if fuga or nombra or detalles:
                c.extra["fuga"] = fuga + detalles
                c.notas = ("ATENCIÓN: el borrador aún copia frases, nombra el tipo o trae fechas o cifras; "
                           "reescríbelo. " + c.notas)
        log(f"   {c.id}: {c.consulta[:100]}{'  ⚠️' if c.extra.get('fuga') else ''}")
        salida.append(c)
    return salida


def resumen_sentencia(texto: str, tipo: str, fecha: str) -> str:
    """Resumen de 3 líneas, sin LLM, para que el anotador ubique la sentencia sin leerla completa:
    etiqueta del portal y fecha; inicio del texto; inicio de los resolutivos (si se encuentran)."""
    inicio = re.sub(r"\s+", " ", texto[:260]).strip()
    mitad = len(texto) // 2
    marcas = list(_RE_RESOLUTIVOS.finditer(texto, mitad))
    if marcas:
        final = "Resolutivos: " + re.sub(r"\s+", " ", texto[marcas[-1].start():marcas[-1].start() + 260]).strip()
    else:  # sin marca, el final del texto (tras quitar el anexo de testado suelen ser los resolutivos)
        final = "Final del texto: …" + re.sub(r"\s+", " ", texto[-260:]).strip()
    return f"{etiqueta_tipo(tipo)} · publicada {fecha}\nInicio: {inicio}…\n{final}…"


def hoja_juicios(store, clasificador, consultas: list[Consulta], conceptos: dict[str, str], n_consultas: int = 30,
                 k: int = 10, semilla: int = config.EVAL_SEED) -> pd.DataFrame:
    """Hoja de juicios de relevancia: para `n_consultas` consultas (repartidas entre familias), los `k` primeros
    candidatos que ve la app, en orden aleatorio (el anotador no debe ver el rango ni el puntaje)."""
    from ..embeddings import embed_query
    from ..pipeline import combinar_candidatos

    rng = np.random.default_rng(semilla)
    elegibles = sorted([c for c in consultas if c.consulta.strip() and not c.sin_respuesta],
                       key=lambda c: (c.familia or "", c.id))
    paso = max(1, len(elegibles) // max(1, n_consultas))
    elegidas = elegibles[::paso][:n_consultas]
    corpus = store.corpus
    filas = []
    for c in elegidas:
        q = embed_query(c.consulta)
        lexica = f"{c.consulta} {conceptos.get(c.id, '')}"
        amplios = store.buscar(c.consulta, query_embedding=q, consulta_lexica=lexica)
        tabla = clasificador.clasificar(c.consulta, conceptos.get(c.id, c.consulta), q, recuperados=amplios).tabla
        filtro = [t for cat in tabla["tipo"].head(config.NUM_CATEGORIAS_FILTRO) for t in clasificador.tipos_de(cat)]
        estrictos = store.buscar(c.consulta, tipos=filtro, query_embedding=q, consulta_lexica=lexica)
        ids = [x.doc_id for x, _ in combinar_candidatos(estrictos, amplios, config.RAG_NUM_RESULTS_DESIRED)[0][:k]]
        for d in rng.permutation(ids):
            r = corpus.loc[d]
            filas.append({"consulta_id": c.id, "consulta": c.consulta, "doc_id": d,
                          "resumen": resumen_sentencia(r["texto"], r["tipo"], r["fecha"]),
                          "relevancia": "", "anotador": ""})
    return pd.DataFrame(filas)


def hoja_fichas(corpus: pd.DataFrame, familias: dict[str, str], n: int = 40,
                semilla: int = config.EVAL_SEED) -> pd.DataFrame:
    """Hoja del conjunto C: `n` sentencias penales repartidas entre familias (por turnos), con las columnas que
    el abogado llena leyendo la sentencia."""
    rng = np.random.default_rng(semilla)
    penal = corpus[corpus["materia"] == "penal"].assign(familia=lambda d: d["tipo"].map(familias))
    grupos = {f: list(rng.permutation(g["id"].tolist())) for f, g in penal.groupby("familia")}
    orden = sorted(grupos, key=lambda f: -len(grupos[f]))
    elegidos: list[str] = []
    while len(elegidos) < n and any(grupos.values()):
        for f in orden:
            if grupos[f] and len(elegidos) < n:
                elegidos.append(grupos[f].pop())
    from ..fichas_precalculadas import ficha_guardada, version_ficha

    settings = config.LLMSettings()
    filas = []
    for d in elegidos:
        r = penal.set_index("id").loc[d]
        ficha = ficha_guardada(d, r["texto"], settings)
        if ficha is not None:  # lo mismo que muestra la app en "Ficha completa"
            from ..fichas import verificaciones

            v = verificaciones(ficha.campos_markdown(), r["texto"])
            ficha_app = ficha.informe_markdown(v["articulos_no_encontrados"], v["sentido_reglas"])
        filas.append({"doc_id": d, "familia": r["familia"], "resumen": resumen_sentencia(r["texto"], r["tipo"], r["fecha"]),
                      "sentido_fallo": "", "etapa": "", "acto_impugnado": "", "delito": "", "articulos": "",
                      "agravios": "", "consulta_id": "", "ficha_aceptable": "", "errores_ficha": "",
                      "version_ficha": version_ficha(settings) if ficha is not None else "",
                      "ficha_app": ficha_app if ficha is not None else ""})
    return pd.DataFrame(filas)
