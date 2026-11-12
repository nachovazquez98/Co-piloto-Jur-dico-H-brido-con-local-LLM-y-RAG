"""Regresión con consultas fijas (sin LLM: los conceptos clave están precalculados).

Dos conjuntos:
- `EJEMPLOS_V10`: las 18 consultas multimateria del notebook, con las que se midió el "15 de 18" de v10
  (Hit@3 de categoría = 0.83, IC 95 % [0.61, 0.94], n = 18, no independiente del ajuste). Cada una lleva el patrón
  de tipos que cuenta como acierto, tomado de su propia etiqueta; None = tema sin tipo en el corpus.
- Los 16 ejemplos penales de la app (`ejemplos.EJEMPLOS_PENALES`), evaluados por familia.

No es una evaluación (las consultas no son independientes ni representativas): sirve para detectar si un cambio
rompe casos que funcionaban y para decisiones de diseño rápidas. Los conceptos se guardan en
`evaluacion/regresion_conceptos.json` (comando `regresion --recalcular-conceptos`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..ejemplos import EJEMPLOS_PENALES, Ejemplo

CONCEPTOS_PATH = config.EVAL_DIR / "regresion_conceptos.json"


@dataclass(frozen=True)
class CasoRegresion:
    ejemplo: Ejemplo
    patron_esperado: str | None   # regex sobre el tipo; None = sin tipo en el corpus


EJEMPLOS_V10: list[Ejemplo] = [
    Ejemplo("Penal", "Robo calificado",
            "Se acusa a un individuo de robo calificado por haber sustraído con violencia un teléfono móvil en la vía pública durante la noche. La víctima lo identificó plenamente y existen cámaras de seguridad que registraron el hecho."),
    Ejemplo("Penal", "Lesiones dolosas (riña)",
            "Dos personas riñeron en un bar, resultando una de ellas con lesiones que tardan más de 15 días en sanar. Se imputa el delito de lesiones dolosas al agresor principal, quien argumenta legítima defensa."),
    Ejemplo("Penal", "Homicidio culposo",
            "Un conductor, presuntamente bajo los efectos del alcohol, atropelló a un peatón causándole la muerte. Se le imputa homicidio culposo agravado. La defensa argumenta que el peatón cruzó imprudentemente."),
    Ejemplo("Penal", "Fraude",
            "Se denuncia a una persona por fraude, ya que solicitó un anticipo para realizar una obra de construcción y desapareció sin iniciar los trabajos ni devolver el dinero."),
    Ejemplo("Penal", "Justicia para adolescentes (robo)",
            "Un adolescente es señalado por sustraer mercancía de una tienda departamental. Se busca determinar la medida socioeducativa aplicable conforme a la Ley Nacional del Sistema Integral de Justicia Penal para Adolescentes."),
    Ejemplo("Penal", "Orden de aprehensión (amparo)",
            "Se promueve amparo indirecto contra una orden de aprehensión por considerarla violatoria de derechos fundamentales, argumentando falta de fundamentación y motivación."),
    Ejemplo("Mercantil", "Acción cambiaria directa (pagaré)",
            "Se ejercita acción cambiaria directa por falta de pago de un pagaré vencido. El demandado se niega a pagar alegando que su firma fue falsificada."),
    Ejemplo("Mercantil", "Incumplimiento de contrato de suministro",
            "Una empresa demanda a otra por incumplimiento de contrato de suministro de mercancías, ya que la entrega se realizó fuera de plazo y con productos defectuosos. Se reclaman daños y perjuicios."),
    Ejemplo("Mercantil", "Pago de cheque devuelto",
            "Se demanda el pago de un cheque que fue devuelto por insuficiencia de fondos. El librador argumenta que el cheque fue robado y llenado sin su consentimiento."),
    Ejemplo("Civil", "Rescisión de contrato de arrendamiento",
            "Se demanda la rescisión de un contrato de arrendamiento por falta de pago de rentas por más de tres meses. El inquilino argumenta problemas económicos imprevistos debido a la pérdida de su empleo y solicita una prórroga."),
    Ejemplo("Civil", "Acción reivindicatoria",
            "El propietario de un terreno demanda la acción reivindicatoria contra el poseedor actual, quien construyó una barda invadiendo parte de su propiedad. El demandado alega prescripción adquisitiva."),
    Ejemplo("Civil", "Daños y perjuicios (accidente vehicular)",
            "Se demanda el pago de daños y perjuicios por un accidente vehicular causado por negligencia. El vehículo del demandante sufrió pérdida total. El demandado niega su responsabilidad."),
    Ejemplo("Civil", "Rescisión de compraventa (vicios ocultos)",
            "Un comprador demanda la rescisión de un contrato de compraventa de un inmueble debido a vicios ocultos (humedades estructurales) no informados por el vendedor."),
    Ejemplo("Civil", "Información ad perpetuam / usucapión",
            "Se promueven diligencias de información ad perpetuam para acreditar la posesión de un inmueble no inscrito en el Registro Público, con el fin de obtener un título de propiedad por prescripción."),
    Ejemplo("Familiar", "Divorcio",
            "Se solicita el divorcio necesario por causal de abandono del hogar conyugal por más de seis meses sin causa justificada. Se pide la disolución del vínculo matrimonial y la liquidación de la sociedad conyugal."),
    Ejemplo("Familiar", "Aumento de pensión alimenticia",
            "Se demanda el aumento de la pensión alimenticia para dos menores de edad, argumentando un incremento en las necesidades escolares y médicas, y una mejora en la capacidad económica del deudor alimentario."),
    Ejemplo("Familiar", "Modificación de guarda y custodia",
            "Un progenitor solicita la modificación de la guarda y custodia de un menor, alegando que el otro progenitor expone al niño a un ambiente inadecuado y negligencia en sus cuidados."),
    Ejemplo("Familiar", "Régimen de convivencias",
            "Se demanda la regulación del régimen de convivencias con un menor, ya que el progenitor custodio impide u obstaculiza el contacto del niño con el progenitor no custodio."),
]

_PATRONES_V10 = [
    r"^penal/robo-calificado", r"^penal/lesiones", r"^penal/homicidio.*culp", r"^penal/fraude", None, None,
    r"^mercantil/accion-cambiaria", None, None, r"arrendamiento", r"reivindicatoria", r"danos-y-perjuicios",
    r"compraventa", r"perpetuam|usucapion|prescripcion-positiva", r"divorcio", r"alimentos|pension",
    r"custodia", r"convivenc|visitas",
]
CASOS_V10 = [CasoRegresion(e, p) for e, p in zip(EJEMPLOS_V10, _PATRONES_V10)]


def acierta(tipos_top: list[str], patron: str | None) -> bool | None:
    """True si algún tipo del top coincide con el patrón esperado; None si el caso no tiene tipo en el corpus."""
    if patron is None:
        return None
    return any(re.search(patron, t) for t in tipos_top)


def clave(texto: str, version: str) -> str:
    """Clave de caché: texto + texto completo del prompt (si se edita el prompt, se recalcula)."""
    import hashlib

    from .. import prompts

    return hashlib.sha256(f"{prompts.prompt_conceptos(version)}|{texto}".encode()).hexdigest()[:16]


def leer_conceptos(path: Path = CONCEPTOS_PATH) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def conceptos_para(texto: str, version: str, cache: dict[str, str]) -> str:
    """Conceptos precalculados; si faltan, la consulta completa (lo que hace la app si el LLM falla)."""
    return cache.get(clave(texto, version), texto)


def calcular_conceptos(llm, path: Path = CONCEPTOS_PATH, log=print) -> dict[str, str]:
    """Precalcula con el LLM los conceptos de las 18 consultas v10 (prompt v10, como se midieron) y de los 16
    ejemplos penales (prompts v10 y v11)."""
    from .. import prompts
    from ..pipeline import limpiar_conceptos

    cache = leer_conceptos(path)
    trabajos = [(c.ejemplo.texto, "v10") for c in CASOS_V10]
    trabajos += [(e.texto, v) for e in EJEMPLOS_PENALES for v in ("v10", "v11")]
    for texto, version in trabajos:
        k = clave(texto, version)
        if k not in cache:
            resp = llm.generate(prompts.prompt_conceptos(version).format(texto_consulta=texto))
            cache[k] = limpiar_conceptos(resp, texto)[0]
            log(f"   [{version}] {cache[k][:90]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return cache


# Variantes del clasificador para documentar las decisiones de v11 (Jaccard/KEM, lista de tokens, familias).
def variantes():
    from ..clasificador import PesosClasificador as P

    return [
        ("v10: Jaccard 0.20 + KEM 0.20, lista de tokens", P(0.20, 0.20, 0.20, 0.05, 0.35, True, True), False),
        ("Jaccard 0.40 sin KEM, lista de tokens", P(0.20, 0.40, 0.00, 0.05, 0.35, True, True), False),
        ("Jaccard 0.40 sin KEM, sin lista", P(0.20, 0.40, 0.00, 0.05, 0.35, True, False), False),
        ("Jaccard 0.40 sin KEM, sin lista, familias enriquecidas", P(0.20, 0.40, 0.00, 0.05, 0.35, True, False), True),
        ("Configuración actual (config.py)", P.desde_config(), None),
    ]


def _top3(store, clasif, texto, conceptos, materias, pesos, embed_query):
    from .experimentos import ranking_categorias

    q = embed_query(texto)
    amplios = store.buscar(texto, query_embedding=q, consulta_lexica=f"{texto} {conceptos}", materias=materias)
    comp, _ = clasif.componentes(texto, conceptos, q, recuperados=amplios)
    return ranking_categorias(comp, pesos)[:3]


def correr_regresion(store, log=print):
    """Hit@3 de cada variante en las 18 consultas v10 (tipo, todas las materias, conceptos v10) y en los 16
    ejemplos penales (familia, solo penal, conceptos v11). Devuelve la corrida con resumen y detalle."""
    import pandas as pd

    from ..clasificador import ClasificadorHibrido
    from ..embeddings import embed_query
    from .familias import cargar_familias, etiqueta_familia
    from .experimentos import familia_de_categoria
    from .reporte import Corrida

    cache = leer_conceptos()
    if not cache:
        raise FileNotFoundError(f"Faltan los conceptos precalculados ({CONCEPTOS_PATH}). Corre "
                                "`python -m copiloto_juridico.evaluacion regresion --recalcular-conceptos`.")
    tipos = store.corpus["tipo"].unique().tolist()
    penales = [t for t in tipos if t.startswith("penal/")]
    familias = cargar_familias(tipos)
    log("Clasificadores: tipo (todas las materias) y familia (penal, con y sin enriquecer)...")
    clas_tipo = ClasificadorHibrido(tipos, "tipo")
    clas_fam = {e: ClasificadorHibrido(penales, "familia", familias, enriquecer=e) for e in (False, True)}

    resumen, detalle = [], []
    for nombre, pesos, enriquecer in variantes():
        log(f"   {nombre}")
        cf = clas_fam[config.ENRIQUECER_FAMILIAS if enriquecer is None else enriquecer]
        hits_v10, hits_pen = [], []
        for c in CASOS_V10:
            top = _top3(store, clas_tipo, c.ejemplo.texto, conceptos_para(c.ejemplo.texto, "v10", cache),
                        config.MATERIAS, pesos, embed_query)
            h = acierta(top, c.patron_esperado)
            if h is not None:
                hits_v10.append(h)
            detalle.append({"conjunto": "v10 (18)", "variante": nombre, "ejemplo": c.ejemplo.tipo,
                            "acierto": "—" if h is None else ("sí" if h else "no"), "top-3": " · ".join(top)})
        for e in EJEMPLOS_PENALES:
            top = _top3(store, cf, e.texto, conceptos_para(e.texto, "v11", cache), ["penal"], pesos, embed_query)
            fams = [familia_de_categoria(t, familias) for t in top]
            h = None if e.sin_tipo_en_corpus else familias[e.tipo_corpus] in fams
            if h is not None:
                hits_pen.append(h)
            detalle.append({"conjunto": "penal (16)", "variante": nombre, "ejemplo": e.etiqueta,
                            "acierto": "—" if h is None else ("sí" if h else "no"),
                            "top-3": " · ".join(etiqueta_familia(f) for f in fams)})
        resumen.append({"variante": nombre, "Hit@3 tipo (v10, 18)": f"{sum(hits_v10)}/{len(hits_v10)}",
                        "Hit@3 familia (penal, 16)": f"{sum(hits_pen)}/{len(hits_pen)}"})
    corrida = Corrida("regresion", parametros={"conceptos_v10": "v10", "conceptos_penal": "v11"})
    corrida.tabla("Resumen por variante",
                  "Aciertos en el top-3 de categorías (sin LLM; conceptos precalculados). Los temas sin tipo en el "
                  "corpus no cuentan. No es una evaluación: son consultas de ejemplo, no independientes, y se usaron "
                  "para decidir la configuración.", pd.DataFrame(resumen))
    corrida.tabla("Detalle por ejemplo", "Top-3 de cada ejemplo y variante.", pd.DataFrame(detalle))
    corrida.extra["aciertos"] = {r["variante"]: r for r in resumen}
    return corrida
