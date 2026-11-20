"""Corpus versionado: texto limpio de las sentencias en Parquet (comprimido, en fragmentos < 50 MB)."""

from __future__ import annotations

import hashlib
import logging
import re
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


_RE_OMITIDAS = r"págs? omitidas?"


def _compacto(texto: str) -> str:
    """Solo letras y números en minúsculas: la extracción del PDF parte palabras ("fracció n") de forma distinta."""
    return re.sub(r"[^0-9a-zñáéíóúü]", "", texto.lower())


def contenido_en(viejo: str, nuevo: str, ventanas: int = 20, largo: int = 60) -> float:
    """Fracción de ventanas repartidas del texto viejo que aparecen en el nuevo. Un PDF vuelto a descargar de la
    misma sentencia contiene el texto muestreado que ya había (en las 360 redescargas: ≥ 0.80 o ≤ 0.15)."""
    a, b = _compacto(viejo), _compacto(nuevo)
    if len(a) < largo:
        return 0.0
    paso = max(1, (len(a) - largo) // ventanas)
    muestras = [a[i:i + largo] for i in range(0, len(a) - largo, paso)][:ventanas]
    return sum(m in b for m in muestras) / len(muestras)


def descargas_ajenas(actual: pd.DataFrame, nuevos: pd.DataFrame, minimo: float = 0.5) -> pd.DataFrame:
    """PDFs nuevos con el id de una sentencia que ya está en el corpus pero con el texto de otra (se guardó otro
    documento con ese nombre). No deben reemplazarla."""
    previo = actual.set_index("id")["texto"]
    filas = [{"id": r.id, "contenido": round(contenido_en(previo[r.id], r.texto), 2)}
             for r in nuevos.itertuples() if r.id in previo.index]
    t = pd.DataFrame(filas, columns=["id", "contenido"])
    return t[t["contenido"] < minimo]


def duplicados_de_texto(corpus: pd.DataFrame, preferidos: set[str] | None = None) -> pd.DataFrame:
    """Sentencias publicadas dos veces en el portal con distinto nombre (mismo texto). Devuelve (id que se quita,
    id que se queda); se queda el de `preferidos` (p. ej. fuente de consultas) o, si no, el primero por fecha."""
    preferidos = preferidos or set()
    huella = corpus["texto"].map(lambda t: hashlib.sha1((_compacto(t[:6000]) + _compacto(t[-3000:])).encode()).hexdigest())
    orden = corpus.assign(_h=huella, _p=~corpus["id"].isin(preferidos)).sort_values(["_p", "fecha", "id"])
    filas = []
    for _, g in orden.groupby("_h", sort=False):
        filas += [{"id": d, "igual_a": g["id"].iloc[0]} for d in g["id"].iloc[1:]]
    return pd.DataFrame(filas, columns=["id", "igual_a"])


def unir_fuentes(fuentes: list[pd.DataFrame]) -> pd.DataFrame:
    """Une corpus de varias fuentes sin repetir sentencias. Si una sentencia aparece más de una vez, gana la versión
    completa (sin "págs omitidas": p. ej. el PDF vuelto a descargar) y, entre iguales, la de la primera fuente
    (corpus actual > ChromaDB > PDFs)."""
    todo = pd.concat(fuentes, ignore_index=True)
    incompleta = todo["texto"].str.contains(_RE_OMITIDAS, regex=True)
    todo = todo.assign(_incompleta=incompleta).sort_values("_incompleta", kind="stable")
    return todo.drop_duplicates(subset="id", keep="first").drop(columns="_incompleta").sort_index()


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
