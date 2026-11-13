"""Re-ranking opcional con un cross-encoder (v11.1, experimento; apagado por defecto: config.RERANKER = None).

El cross-encoder lee la consulta junto con el fragmento más parecido de cada candidato y les da un puntaje de
relevancia; reordena solo los candidatos de la búsqueda (hasta 30). Se activa por defecto únicamente si, con los
juicios humanos, la diferencia pareada en nDCG@5 contra la app sin re-ranker es positiva y su IC 95 % no incluye 0.
"""

from __future__ import annotations

from functools import lru_cache

from . import config


def _dispositivo() -> str:
    """GPU solo si quedan ~2 GB libres junto a Ollama; si no, CPU (con 30 candidatos es tolerable)."""
    if config.RERANKER_DISPOSITIVO != "auto":
        return config.RERANKER_DISPOSITIVO
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2.0e9:
            return "cuda"
    except Exception:  # sin CUDA o sin driver: CPU
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _modelo(nombre: str, dispositivo: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(nombre, device=dispositivo, max_length=512)


def reordenar(consulta: str, candidatos: list, fragmento, nombre: str | None = None,
              max_candidatos: int = config.RAG_NUM_DOCS_TO_FETCH) -> list:
    """Reordena `candidatos` (tuplas (Candidato, fuente)) por el puntaje del cross-encoder sobre
    (consulta, fragmento(doc_id)). Los que pasan de `max_candidatos` quedan al final en su orden."""
    nombre = nombre or config.RERANKER
    if not nombre or not candidatos:
        return candidatos
    cabeza, cola = candidatos[:max_candidatos], candidatos[max_candidatos:]
    puntajes = _modelo(nombre, _dispositivo()).predict([(consulta, fragmento(c.doc_id)) for c, _ in cabeza],
                                                       batch_size=8)
    orden = sorted(range(len(cabeza)), key=lambda i: -float(puntajes[i]))
    return [cabeza[i] for i in orden] + cola
