"""Fase 2: generación del índice de embeddings (fragmentos de cada sentencia).

Los embeddings se guardan en el repo (float16) para que la app arranque sin GPU y sin reindexar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..embeddings import chunks_de, embed_texts

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
_SHARD_ROWS = 15000


def corpus_fingerprint(corpus: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for doc_id, texto in zip(corpus["id"], corpus["texto"]):
        h.update(doc_id.encode())
        h.update(str(len(texto)).encode())
    return h.hexdigest()[:16]


def index_params() -> dict:
    return {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "chunk_size": config.CHUNK_SIZE_CHARS,
        "chunk_overlap": config.CHUNK_OVERLAP_CHARS,
        "max_chunks_per_doc": config.MAX_CHUNKS_PER_DOC,
        "max_chunks_penal": config.MAX_CHUNKS_PENAL,
    }


def build_chunks(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id, texto, materia in zip(corpus["id"], corpus["texto"], corpus["materia"]):
        for n, chunk in enumerate(chunks_de(texto, materia)):
            rows.append({"chunk_id": f"{doc_id}#{n}", "doc_id": doc_id, "chunk_idx": n, "texto": chunk})
    return pd.DataFrame(rows)


def build_index(corpus: pd.DataFrame, directory: Path = config.INDICE_DIR,
                log: Callable[[str], None] = print) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(corpus)
    log(f"Generando embeddings de {len(chunks)} fragmentos de {len(corpus)} sentencias "
        f"con '{config.EMBEDDING_MODEL_NAME}'...")
    t0 = time.time()
    parts = []
    lote = 2000
    for start in range(0, len(chunks), lote):
        parts.append(embed_texts(chunks["texto"].iloc[start:start + lote].tolist(), kind="passage"))
        log(f"   -> {min(start + lote, len(chunks))}/{len(chunks)} fragmentos ({time.time() - t0:.0f}s)")
    emb = np.vstack(parts).astype(np.float16)
    manifest = {**index_params(), "corpus_fingerprint": corpus_fingerprint(corpus), "num_docs": len(corpus),
                "num_chunks": len(chunks), "dim": int(emb.shape[1])}
    _guardar(chunks, emb, manifest, directory)
    log(f"Índice guardado en '{directory}' ({time.time() - t0:.0f}s).")
    return manifest


def _guardar(chunks: pd.DataFrame, emb: np.ndarray, manifest: dict, directory: Path) -> None:
    for old in list(directory.glob("embeddings-*.npy")) + list(directory.glob("fragmentos.parquet")):
        old.unlink()
    chunks[["chunk_id", "doc_id", "chunk_idx"]].to_parquet(directory / "fragmentos.parquet", compression="zstd",
                                                           index=False)
    for n, start in enumerate(range(0, len(emb), _SHARD_ROWS)):
        np.save(directory / f"embeddings-{n:03d}.npy", emb[start:start + _SHARD_ROWS])
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def recortar_indice(corpus: pd.DataFrame, directory: Path = config.INDICE_DIR,
                    log: Callable[[str], None] = print) -> dict:
    """Conserva del índice actual solo los fragmentos de las sentencias del corpus, sin recalcular embeddings
    (cuando el corpus se reduce, como al dejarlo solo penal). Exige los mismos parámetros y, por sentencia, el mismo número de fragmentos que
    produciría `build_chunks`; si no, hay que reindexar."""
    manifest, chunks, emb = load_index(directory)
    if {k: manifest.get(k) for k in index_params()} != index_params():
        raise ValueError("El índice se construyó con otros parámetros: corre `indexar` sin --recortar.")
    esperado = build_chunks(corpus).groupby("doc_id").size()
    actual = chunks.groupby("doc_id").size().reindex(esperado.index)
    distintos = esperado.index[actual.ne(esperado)].tolist()
    if distintos:
        raise ValueError(f"{len(distintos)} sentencias sin sus fragmentos en el índice actual (p. ej. {distintos[0]}): "
                         "corre `indexar` sin --recortar.")
    quedan = chunks["doc_id"].isin(set(corpus["id"])).to_numpy()
    chunks, emb = chunks[quedan].reset_index(drop=True), emb[quedan]
    manifest = {**index_params(), "corpus_fingerprint": corpus_fingerprint(corpus), "num_docs": len(corpus),
                "num_chunks": len(chunks), "dim": int(emb.shape[1])}
    _guardar(chunks, emb, manifest, directory)
    log(f"Índice recortado: {int(quedan.sum()):,} de {len(quedan):,} fragmentos, {len(corpus):,} sentencias.")
    return manifest


def load_index(directory: Path = config.INDICE_DIR) -> tuple[dict, pd.DataFrame, np.ndarray]:
    manifest = json.loads((directory / MANIFEST).read_text())
    chunks = pd.read_parquet(directory / "fragmentos.parquet")
    emb = np.vstack([np.load(p) for p in sorted(directory.glob("embeddings-*.npy"))])
    return manifest, chunks, emb
