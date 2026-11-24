"""Esfuerzo simulado: sustituye al estudio de uso con abogados, que no están disponibles.

¿Cuántas sentencias tendría que abrir alguien para encontrar `K` precedentes útiles? Útil = referencia automática de
relevancia 2: misma familia de delito (categoría del STJ) y mismo acto impugnado (mayoría de lectores, ver
`referencias.py`) que la sentencia de la que salió la consulta, sin contar esa sentencia. Se compara:

- portal por delito: todas las sentencias de la misma categoría del STJ, de la más reciente a la más antigua (como
  lista el portal). Optimista para el portal: supone que se elige exactamente la categoría correcta.
- portal por delito y tipo de sentencia: además, el filtro "Definitiva / Interlocutoria" del portal.
- búsqueda por palabras: BM25 con el texto de la consulta.
- la app: la lista de candidatos de `buscar_candidatos`, en su orden.
- la app con el delito elegido: el usuario elige en la app la familia de delito de su caso (la de la categoría
  del STJ, igual de optimista que el portal) y la búsqueda se filtra por ella en lugar de por el clasificador.

Si no se llega a `K` dentro de las primeras `TOPE` sentencias, cuenta como `TOPE` + 1 (no encontró). Advertencia: la
app ordena por familia y acto impugnado, que es parte de esta definición de "útil"; el resultado mide cuánto ahorra
ese orden, no si un abogado consideraría útiles esos precedentes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..metadatos import NO_DETERMINADO

K, TOPE = 3, 30
ESTRATEGIAS = ("portal: delito", "portal: delito + tipo de sentencia", "búsqueda por palabras", "app",
               "app con el delito elegido")


def tipo_sentencia(acto: str) -> str:
    """El filtro del portal: la sentencia definitiva es "Definitiva"; los autos (vinculación, medidas...),
    "Interlocutoria"."""
    return "Definitiva" if acto == "sentencia definitiva" else "Interlocutoria"


def abiertas_hasta(lista: list[str], utiles: set[str], k: int = K, tope: int = TOPE) -> int:
    """Posición de la k-ésima sentencia útil en la lista (sentencias que hay que abrir); tope + 1 si no se llega."""
    n = 0
    for i, d in enumerate(lista[:tope], 1):
        n += d in utiles
        if n == k:
            return i
    return tope + 1


def actos_de_referencia() -> tuple[dict[str, str], str]:
    """Acto impugnado por sentencia: el consenso de `referencias` si existe; si no, el de las reglas."""
    from .. import metadatos as md
    from .referencias import CONSENSO, REFERENCIAS_PATH

    if REFERENCIAS_PATH.exists():
        t = pd.read_csv(REFERENCIAS_PATH, dtype=str)
        return dict(zip(t["doc_id"], t[f"acto_impugnado:{CONSENSO}"].fillna(NO_DETERMINADO))), "mayoría de lectores"
    return {d: m["acto_impugnado"] for d, m in md.cargar().items()}, "solo reglas (falta `referencias`)"


def experimento_esfuerzo(env, log: Callable[[str], None] = print):
    from ..pipeline import buscar_candidatos
    from .experimentos import _fila_diferencia, preparar

    corrida = env.corrida
    actos, fuente = actos_de_referencia()
    corrida.parametros.update({"k": K, "tope": TOPE, "acto_impugnado": fuente})
    if fuente.startswith("solo reglas"):
        corrida.advertir("Sin referencias automáticas: el acto impugnado sale solo de las reglas, que son las mismas que "
                         "usa la app para ordenar (favorece a la app).")
    corpus = env.store.corpus
    fam = corpus["tipo"].map(env.familias)
    por_fecha = corpus.sort_values("fecha", ascending=False, kind="stable")
    filas, sin_acto = [], 0
    for p in preparar(env):
        c = p.consulta
        if not c.documento_conocido:
            continue
        acto = actos.get(c.doc_id_fuente, NO_DETERMINADO)
        if acto in (NO_DETERMINADO, "otro"):
            sin_acto += 1
            continue
        utiles = set(corpus.index[(fam == c.familia) & corpus["id"].map(lambda d: actos.get(d) == acto)]) - {
            c.doc_id_fuente}
        if len(utiles) < K:
            continue
        portal = [d for d in por_fecha.index[por_fecha["tipo"] == c.tipo] if d != c.doc_id_fuente]
        listas = {
            "portal: delito": portal,
            "portal: delito + tipo de sentencia": [d for d in portal if tipo_sentencia(actos.get(d, "")) ==
                                                   tipo_sentencia(acto)],
            "búsqueda por palabras": [x.doc_id for x in env.store.buscar(c.consulta, modo="lexica", num_docs=TOPE + 1,
                                                                        consulta_lexica=p.lexica,
                                                                        excluir={c.doc_id_fuente})],
            **{e: [x.doc_id for x, _ in buscar_candidatos(env.store, env.clasificador, c.consulta, p.conceptos, p.q,
                                                         config.RAG_NUM_RESULTS_DESIRED, familias=f).candidatos
                   if x.doc_id != c.doc_id_fuente]
               for e, f in (("app", ()), ("app con el delito elegido", (c.familia,)))},
        }
        filas.append({"id": c.id, "útiles en el corpus": len(utiles),
                      **{e: abiertas_hasta(listas[e], utiles) for e in ESTRATEGIAS}})
    log(f"Esfuerzo simulado: {len(filas)} consultas ({sin_acto} sin acto impugnado de referencia).")
    if not filas:
        corrida.advertir("Ninguna consulta con acto impugnado de referencia y al menos 3 sentencias útiles.")
        return corrida
    t = pd.DataFrame(filas)
    resumen = []
    for e in ESTRATEGIAS:
        x = t[e].to_numpy()
        resumen.append({"estrategia": e, "consultas": len(x), "mediana de sentencias abiertas": float(np.median(x)),
                        "promedio": float(np.mean(x)), f"encontró {K} en {TOPE}": float(np.mean(x <= TOPE))})
    corrida.tabla(f"Sentencias que hay que abrir para encontrar {K} útiles",
                  f"Útil = misma familia (categoría del STJ) y mismo acto impugnado ({fuente}) que la sentencia de la "
                  f"consulta, sin contarla. Si no se llega en {TOPE}, cuenta {TOPE + 1}. El portal supone que se elige "
                  "la categoría exacta y 'app con el delito elegido', que se elige su familia (optimistas).",
                  pd.DataFrame(resumen))
    # con signo invertido: Δ = sentencias que ahorra la app y "gana A" = la app abre menos
    pares = [("app", e) for e in ESTRATEGIAS[:3]] + [("app con el delito elegido", "app")]
    comp = [{"A": a, "B": b, "métrica": "sentencias que ahorra A", "n": len(t),
             **_fila_diferencia((-t[a]).tolist(), (-t[b]).tolist())} for a, b in pares]
    corrida.tabla("Comparaciones pareadas", "Δ = media(B − A) por consulta: sentencias que A ahorra (positivo = A abre "
                  "menos). La última fila mide cuánto ayuda elegir el delito en la app. IC95 bootstrap pareado.",
                  pd.DataFrame(comp))
    corrida.tabla("Por consulta", "Sentencias abiertas con cada estrategia.", t, diagnostico=True)
    return corrida
