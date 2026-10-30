"""Base vectorial (ChromaDB) construida automáticamente a partir del índice versionado.

La carpeta `data/chroma` no se sube a git: al arrancar se compara la huella del índice y, si cambió o no
existe, se reconstruye en ~1 minuto con los embeddings precalculados (no hace falta GPU).
"""

from __future__ import annotations

import json
import logging
import shutil
import unicodedata
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .datos.corpus import load_corpus
from .datos.indice import load_index
from .embeddings import embed_query

logger = logging.getLogger(__name__)

_FINGERPRINT_FILE = "copiloto_fingerprint.json"


@dataclass
class Candidato:
    doc_id: str
    materia: str
    delito_o_accion: str
    tipo: str
    fecha: str
    similitud: float      # coseno del fragmento más parecido (embeddings)
    relevancia: float     # puntuación híbrida RRF (embeddings + BM25)
    texto: str


def _normalizar(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode("ascii")


class BM25:
    """BM25 a nivel de sentencia (léxico). Complementa a los embeddings con coincidencias exactas de términos
    jurídicos ("robo calificado", "pagaré", "alimentos")."""

    def __init__(self, textos: list[str], k1: float = 1.5, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        from spacy.lang.es.stop_words import STOP_WORDS

        stop = sorted({_normalizar(w) for w in STOP_WORDS} | {"eliminado", "articulo", "fraccion"})
        self.vectorizer = CountVectorizer(preprocessor=_normalizar, token_pattern=r"(?u)\b[a-z]{3,}\b",
                                          stop_words=stop, min_df=2, max_df=0.5, dtype=np.float32)
        x = self.vectorizer.fit_transform(textos).tocsr()
        dl = np.asarray(x.sum(axis=1)).ravel()
        df = np.asarray((x > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (x.shape[0] - df + 0.5) / (df + 0.5))
        filas = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        tf = x.data
        x.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[filas] / dl.mean())) * idf[x.indices]
        self.matrix = x

    def scores(self, consulta: str) -> np.ndarray:
        q = self.vectorizer.transform([consulta])
        q.data[:] = 1.0
        return (self.matrix @ q.T).toarray().ravel()


class SentenciasStore:
    RRF_K = 60

    def __init__(self, log: Callable[[str], None] = logger.info):
        manifest, chunks, emb = load_index()
        self.manifest = manifest
        self.corpus = load_corpus().set_index("id", drop=False)
        self.chunk_doc = chunks["doc_id"].to_numpy()
        self.emb = emb  # float16, para calcular similitudes de documentos recuperados solo por BM25
        self.collection = self._abrir_chroma(chunks, emb, log)
        log("Construyendo índice léxico BM25...")
        self.doc_ids = self.corpus["id"].to_numpy()
        self.doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.doc_tipos = self.corpus["tipo"].to_numpy()
        self.bm25 = BM25(self.corpus["texto"].tolist())
        log(f"✅ Base lista: {len(self.corpus)} sentencias · {self.collection.count()} fragmentos.")

    def _abrir_chroma(self, chunks: pd.DataFrame, emb: np.ndarray, log: Callable[[str], None]):
        import chromadb

        fingerprint = {k: self.manifest[k] for k in ("corpus_fingerprint", "embedding_model", "num_chunks")}
        fp_path = config.CHROMA_DIR / _FINGERPRINT_FILE
        if fp_path.exists() and json.loads(fp_path.read_text()) == fingerprint:
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            return client.get_collection(config.COLLECTION_NAME)

        log("Construyendo la base vectorial ChromaDB desde el índice incluido en el repositorio (solo la primera vez)...")
        shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        meta_docs = self.corpus[["materia", "delito_o_accion", "tipo", "fecha"]].to_dict("index")
        lote = 5000
        for start in range(0, len(chunks), lote):
            part = chunks.iloc[start:start + lote]
            collection.add(
                ids=part["chunk_id"].tolist(),
                embeddings=emb[start:start + lote].astype(np.float32),
                metadatas=[{"doc_id": d, **meta_docs[d]} for d in part["doc_id"]],
            )
            log(f"   -> {min(start + lote, len(chunks))}/{len(chunks)} fragmentos indexados")
        fp_path.write_text(json.dumps(fingerprint))
        return collection

    # --- Consultas ---
    def tipos(self) -> pd.DataFrame:
        return self.corpus.groupby(["materia", "tipo", "delito_o_accion"]).size().rename("sentencias").reset_index()

    def _densa(self, q: np.ndarray, tipos: list[str] | None) -> dict[str, float]:
        kwargs = {"query_embeddings": [q.tolist()], "n_results": config.RAG_NUM_CHUNKS_TO_FETCH,
                  "include": ["metadatas", "distances"]}
        if tipos:
            kwargs["where"] = {"tipo": {"$in": list(tipos)}}
        res = self.collection.query(**kwargs)
        mejores: dict[str, float] = {}
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            if sim > mejores.get(meta["doc_id"], -1.0):
                mejores[meta["doc_id"]] = sim
        return mejores

    def _similitud_doc(self, doc_id: str, q: np.ndarray) -> float:
        filas = np.flatnonzero(self.chunk_doc == doc_id)
        return float((self.emb[filas].astype(np.float32) @ q).max()) if len(filas) else 0.0

    def buscar(self, consulta: str, num_docs: int = config.RAG_NUM_DOCS_TO_FETCH, tipos: list[str] | None = None,
               query_embedding: np.ndarray | None = None, consulta_lexica: str | None = None) -> list[Candidato]:
        """Búsqueda híbrida: embeddings (fragmentos agrupados por sentencia) + BM25, fusionados con RRF."""
        q = embed_query(consulta) if query_embedding is None else query_embedding
        densos = self._densa(q, tipos)
        rank_denso = {d: r for r, (d, _) in enumerate(sorted(densos.items(), key=lambda kv: kv[1], reverse=True))}

        bm = self.bm25.scores(consulta_lexica or consulta)
        if tipos:
            bm = np.where(np.isin(self.doc_tipos, list(tipos)), bm, 0.0)
        top_bm = [i for i in np.argsort(-bm)[:config.RAG_NUM_DOCS_TO_FETCH * 2] if bm[i] > 0]
        rank_lex = {self.doc_ids[i]: r for r, i in enumerate(top_bm)}

        fusion: dict[str, float] = {}
        for ranking in (rank_denso, rank_lex):
            for d, r in ranking.items():
                fusion[d] = fusion.get(d, 0.0) + 1.0 / (self.RRF_K + r)
        ordenados = sorted(fusion.items(), key=lambda kv: kv[1], reverse=True)[:num_docs]
        out = []
        for doc_id, score in ordenados:
            row = self.corpus.loc[doc_id]
            sim = densos.get(doc_id)
            out.append(Candidato(doc_id=doc_id, materia=row["materia"], delito_o_accion=row["delito_o_accion"],
                                 tipo=row["tipo"], fecha=row["fecha"],
                                 similitud=sim if sim is not None else self._similitud_doc(doc_id, q),
                                 relevancia=score, texto=row["texto"]))
        return out
