"""Corpus versionado: texto limpio de las sentencias en Parquet (comprimido, en fragmentos < 50 MB)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .. import config
from .procesamiento import parse_metadata_from_filename

logger = logging.getLogger(__name__)

CORPUS_COLUMNS = ["id", "fecha", "materia", "delito_o_accion", "tipo", "texto", "origen"]
_SHARD_ROWS = 1000


def save_corpus(df: pd.DataFrame, directory: Path = config.CORPUS_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("sentencias-*.parquet"):
        old.unlink()
    df = df[CORPUS_COLUMNS].sort_values(["materia", "tipo", "fecha", "id"]).reset_index(drop=True)
    paths = []
    for n, start in enumerate(range(0, len(df), _SHARD_ROWS)):
        path = directory / f"sentencias-{n:03d}.parquet"
        df.iloc[start:start + _SHARD_ROWS].to_parquet(path, compression="zstd", compression_level=19, index=False)
        paths.append(path)
    return paths


def load_corpus(directory: Path = config.CORPUS_DIR) -> pd.DataFrame:
    files = sorted(directory.glob("sentencias-*.parquet"))
    if not files:
        return pd.DataFrame(columns=CORPUS_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def import_from_chroma_sqlite(db_path: Path) -> pd.DataFrame:
    """Lee directamente el SQLite de una ChromaDB antigua (p. ej. `chroma_db_juridica_8.2`) y recupera
    el texto limpio y los metadatos de cada sentencia. No necesita la misma versión de chromadb."""
    sqlite_path = Path(db_path) / "chroma.sqlite3"
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT e.embedding_id, doc.string_value
        FROM embeddings e
        JOIN embedding_metadata doc ON doc.id = e.id AND doc.key = 'chroma:document'
        """
    ).fetchall()
    con.close()
    registros = []
    for doc_id, texto in rows:
        meta = parse_metadata_from_filename(doc_id)
        if meta and texto:
            registros.append({"id": doc_id, **meta, "texto": texto, "origen": f"chroma:{Path(db_path).name}"})
    logger.info("Importadas %s sentencias desde %s", len(registros), db_path)
    return pd.DataFrame(registros, columns=CORPUS_COLUMNS)
