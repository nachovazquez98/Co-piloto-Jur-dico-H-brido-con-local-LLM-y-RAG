"""Regresión con consultas fijas (sin LLM: los conceptos clave están precalculados).

Dos conjuntos:
- `EJEMPLOS_V10`: las 6 consultas penales de las 18 del notebook v10 (las de otras materias salieron con el corpus
  solo penal; las corridas anteriores las conservan). Cada una lleva el patrón de tipos que cuenta como acierto;
  None = tema sin tipo en el corpus.
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
]

_PATRONES_V10 = [r"^penal/robo-calificado", r"^penal/lesiones", r"^penal/homicidio.*culp", r"^penal/fraude", None, None]
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
    """Precalcula con el LLM los conceptos de las consultas v10 (prompt v10, como se midieron) y de los 16
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
    """Hit@3 de cada variante en las consultas penales v10 (tipo, conceptos v10) y en los 16
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
    log("Clasificadores: tipo y familia (con y sin enriquecer)...")
    clas_tipo = ClasificadorHibrido(tipos, "tipo")
    clas_fam = {e: ClasificadorHibrido(penales, "familia", familias, enriquecer=e) for e in (False, True)}

    resumen, detalle = [], []
    for nombre, pesos, enriquecer in variantes():
        log(f"   {nombre}")
        cf = clas_fam[config.ENRIQUECER_FAMILIAS if enriquecer is None else enriquecer]
        hits_v10, hits_pen = [], []
        for c in CASOS_V10:
            top = _top3(store, clas_tipo, c.ejemplo.texto, conceptos_para(c.ejemplo.texto, "v10", cache),
                        config.MATERIAS_ACTIVAS, pesos, embed_query)
            h = acierta(top, c.patron_esperado)
            if h is not None:
                hits_v10.append(h)
            detalle.append({"conjunto": f"v10 ({len(CASOS_V10)})", "variante": nombre, "ejemplo": c.ejemplo.tipo,
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
        resumen.append({"variante": nombre, f"Hit@3 tipo (v10, {len(CASOS_V10)})": f"{sum(hits_v10)}/{len(hits_v10)}",
                        "Hit@3 familia (penal, 16)": f"{sum(hits_pen)}/{len(hits_pen)}"})
    log("   Candidatos: orden RRF vs. orden por facetas (familia + acto impugnado)...")
    orden = candidatos_por_orden(store, clas_fam[config.ENRIQUECER_FAMILIAS], familias, cache, embed_query)
    corrida = Corrida("regresion", parametros={"conceptos_v10": "v10", "conceptos_penal": "v11"})
    corrida.tabla("Resumen por variante",
                  "Aciertos en el top-3 de categorías (sin LLM; conceptos precalculados). Los temas sin tipo en el "
                  "corpus no cuentan. No es una evaluación: son consultas de ejemplo, no independientes, y se usaron "
                  "para decidir la configuración.", pd.DataFrame(resumen))
    corrida.tabla("Detalle por ejemplo", "Top-3 de cada ejemplo y variante.", pd.DataFrame(detalle))
    det = pd.DataFrame(orden)
    con = det[det["aciertos rrf"].notna()]
    corrida.tabla("Candidatos: orden RRF vs. orden por facetas",
                  "Configuración actual. Aciertos = de los 3 primeros candidatos (antes de QC_1), cuántos son de la "
                  "familia correcta. 'Pierde' = ejemplos con menos aciertos con el orden por facetas. Acto de la "
                  "consulta y de cada candidato leídos por reglas (metadatos.py).", pd.DataFrame([{
                      "aciertos RRF": f"{int(con['aciertos rrf'].sum())}/{3 * len(con)}",
                      "aciertos facetas": f"{int(con['aciertos facetas'].sum())}/{3 * len(con)}",
                      "ejemplos que pierden": int((con["aciertos facetas"] < con["aciertos rrf"]).sum()),
                      "ejemplos que ganan": int((con["aciertos facetas"] > con["aciertos rrf"]).sum())}]))
    corrida.tabla("Candidatos por ejemplo", "Top-3 de candidatos (familia · acto impugnado) con cada orden.", det)
    corrida.extra["aciertos"] = {r["variante"]: r for r in resumen}
    return corrida


def candidatos_por_orden(store, clasificador, familias: dict[str, str], cache: dict, embed_query, k: int = 3):
    """Top-k de candidatos de cada ejemplo penal con el orden RRF y con el orden por facetas."""
    from ..metadatos import cargar as cargar_metadatos
    from ..pipeline import buscar_candidatos
    from .familias import etiqueta_familia

    actos = {d: m["acto_impugnado"] for d, m in cargar_metadatos().items()}
    filas = []
    for e in EJEMPLOS_PENALES:
        q = embed_query(e.texto)
        conceptos = conceptos_para(e.texto, "v11", cache)
        fila = {"ejemplo": e.etiqueta}
        for orden in ("rrf", "familia_acto"):
            b = buscar_candidatos(store, clasificador, e.texto, conceptos, q, config.RAG_NUM_RESULTS_DESIRED, orden)
            top = [c for c, _ in b.candidatos[:k]]
            fila["acto consulta"] = b.acto_consulta or "—"
            clave = "rrf" if orden == "rrf" else "facetas"
            fila[f"top-{k} {clave}"] = " · ".join(f"{etiqueta_familia(familias.get(c.tipo, c.tipo))} ({actos.get(c.doc_id, '?')})"
                                               for c in top)
            fila[f"aciertos {clave}"] = None if e.sin_tipo_en_corpus else \
                sum(familias.get(c.tipo) == familias[e.tipo_corpus] for c in top)
        filas.append(fila)
    return filas
