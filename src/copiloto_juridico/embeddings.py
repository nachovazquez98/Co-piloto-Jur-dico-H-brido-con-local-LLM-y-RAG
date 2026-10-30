"""Embeddings corregidos.

Respecto al notebook (all-mpnet-base-v2 con promedio sobre todos los tokens, incluido el padding,
y sin normalizar) se corrige:
- Modelo multilingüe `intfloat/multilingual-e5-base` (el corpus está en español).
- Mean pooling con máscara de atención y normalización L2 (sentence-transformers lo hace correctamente).
- Prefijos `query:` / `passage:` que requiere la familia E5.
- Las sentencias se fragmentan (chunks) en lugar de truncarse a los primeros 512 tokens.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _uses_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


@lru_cache(maxsize=2)
def get_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Cargando modelo de embeddings %s en %s...", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    return model


def embed_texts(texts: list[str], kind: str = "passage", model_name: str = config.EMBEDDING_MODEL_NAME,
                batch_size: int = config.EMBEDDING_BATCH_SIZE, show_progress: bool = False) -> np.ndarray:
    """Devuelve una matriz (n, dim) de embeddings normalizados (float32)."""
    if _uses_e5_prefixes(model_name):
        texts = [f"{kind}: {t}" for t in texts]
    model = get_model(model_name)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=show_progress)
    return emb.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text], kind="query")[0]


def chunk_text(text: str, size: int = config.CHUNK_SIZE_CHARS, overlap: int = config.CHUNK_OVERLAP_CHARS,
               max_chunks: int = config.MAX_CHUNKS_PER_DOC) -> list[str]:
    """Divide el texto en fragmentos con traslape y conserva `max_chunks` repartidos
    uniformemente (siempre incluye el inicio y el final, donde están los resolutivos)."""
    text = text.strip()
    if not text:
        return []
    step = max(1, size - overlap)
    chunks = [text[i:i + size] for i in range(0, max(1, len(text) - overlap), step)]
    if len(chunks) > max_chunks:
        idx = np.linspace(0, len(chunks) - 1, num=max_chunks).round().astype(int)
        chunks = [chunks[i] for i in sorted(set(idx))]
    return chunks
