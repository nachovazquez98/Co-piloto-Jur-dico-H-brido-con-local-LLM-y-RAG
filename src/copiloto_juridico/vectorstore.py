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

    @classmethod
    def cargado(cls, textos: list[str], directorio=None, k1: float = 1.5, b: float = 0.75) -> "BM25":
        """El índice guardado en data/cache si el corpus y este código no cambiaron; si no, lo construye (~15 s al
        arrancar la app) y lo guarda. Mismo resultado que `BM25(textos)`: solo evita recalcularlo."""
        import hashlib
        import inspect
        import pickle

        directorio = config.DATA_DIR / "cache" if directorio is None else directorio
        h = hashlib.sha256(f"{inspect.getsource(cls)}\x00{k1}\x00{b}".encode())
        for t in textos:
            h.update(t.encode())
            h.update(b"\x00")
        path = directorio / f"bm25-{h.hexdigest()[:20]}.pkl"
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            pass
        bm25 = cls(textos, k1, b)
        try:
            directorio.mkdir(parents=True, exist_ok=True)
            for viejo in directorio.glob("bm25-*.pkl"):
                viejo.unlink()
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        except OSError:
            pass  # sin permiso de escritura: funciona igual, solo vuelve a construirlo en el siguiente arranque
        return bm25

    def scores(self, consulta: str) -> np.ndarray:
        q = self.vectorizer.transform([consulta])
        q.data[:] = 1.0
        return (self.matrix @ q.T).toarray().ravel()


class SentenciasStore:
    RRF_K = config.RRF_K

    def __init__(self, log: Callable[[str], None] = logger.info):
        manifest, chunks, emb = load_index()
        self.manifest = manifest
        self.corpus = load_corpus().set_index("id", drop=False)
        self.chunk_doc = chunks["doc_id"].to_numpy()
        self.emb = emb  # float16, para calcular similitudes de documentos recuperados solo por BM25
        self.collection = self._abrir_chroma(chunks, emb, log)
        log("Construyendo índice léxico BM25...")
        self.doc_ids = self.corpus["id"].to_numpy()
        self.doc_tipos = self.corpus["tipo"].to_numpy()
        self.doc_materias = self.corpus["materia"].to_numpy()
        self.bm25 = BM25.cargado(self.corpus["texto"].tolist())
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

    def materias_filtro(self, materias: list[str] | None = None) -> list[str] | None:
        """Materias a las que se restringe la búsqueda (config.MATERIAS_ACTIVAS), o None si son todas las del
        corpus (así el comportamiento por defecto es idéntico al de v10)."""
        materias = list(config.MATERIAS_ACTIVAS if materias is None else materias)
        existentes = set(np.unique(self.doc_materias))
        desconocidas = set(materias) - existentes
        if desconocidas:
            # Una materia mal escrita dejaría la búsqueda vacía sin avisar: se falla de forma explícita.
            raise ValueError(f"Materias desconocidas en COPILOTO_MATERIAS: {sorted(desconocidas)}. "
                             f"Válidas: {sorted(existentes)}")
        return None if existentes <= set(materias) else materias

    def _densa(self, q: np.ndarray, tipos: list[str] | None, materias: list[str] | None = None) -> dict[str, float]:
        """Ranking denso: los 300 fragmentos más cercanos (HNSW de ChromaDB) agrupados por sentencia; el puntaje
        de una sentencia es el coseno de su fragmento más parecido. Con `tipos`, ChromaDB filtra por metadatos
        *antes* de buscar vecinos (no es lo mismo que filtrar después)."""
        kwargs = {"query_embeddings": [q.tolist()], "n_results": config.RAG_NUM_CHUNKS_TO_FETCH,
                  "include": ["metadatas", "distances"]}
        if tipos:
            kwargs["where"] = {"tipo": {"$in": list(tipos)}}
        elif materias:
            kwargs["where"] = {"materia": {"$in": list(materias)}}
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

    def fragmento_mas_similar(self, doc_id: str, q: np.ndarray, largo: int = 2500) -> str:
        """Ventana de `largo` caracteres de la sentencia alrededor del fragmento indexado más parecido a la
        consulta. Alternativa a "los primeros 2,500 caracteres" para QC_1 (config.QC1_CONTEXTO)."""
        from .embeddings import chunks_de

        texto = self.corpus.loc[doc_id, "texto"]
        filas = np.flatnonzero(self.chunk_doc == doc_id)
        if not len(filas):
            return texto[:largo]
        mejor = int(np.argmax(self.emb[filas].astype(np.float32) @ q))
        # La misma política de fragmentación con que se construyó el índice (índices de v10 no la guardan: 12).
        chunks = chunks_de(texto, self.corpus.loc[doc_id, "materia"],
                           self.manifest.get("max_chunks_penal", self.manifest.get("max_chunks_per_doc", 12)),
                           self.manifest.get("max_chunks_per_doc", 12))
        inicio = texto.find(chunks[mejor]) if mejor < len(chunks) else -1
        if inicio < 0:
            return texto[:largo]
        inicio = max(0, min(inicio - (largo - len(chunks[mejor])) // 2, len(texto) - largo))
        return texto[inicio:inicio + largo]

    def ranking_lexico(self, consulta_lexica: str, tipos: list[str] | None = None,
                       materias: list[str] | None = None) -> list[str]:
        """Ranking BM25: ids de las 60 sentencias con mayor puntaje (> 0), de mayor a menor."""
        bm = self.bm25.scores(consulta_lexica)
        if tipos:
            bm = np.where(np.isin(self.doc_tipos, list(tipos)), bm, 0.0)
        elif materias:
            bm = np.where(np.isin(self.doc_materias, list(materias)), bm, 0.0)
        return [self.doc_ids[i] for i in np.argsort(-bm)[:config.RAG_NUM_DOCS_TO_FETCH * 2] if bm[i] > 0]

    @staticmethod
    def fusion_rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
        """Reciprocal Rank Fusion: RRF(d) = Σ_listas 1 / (k + rango(d)), con rango empezando en 0.

        Usa solo la posición en cada lista, no los puntajes (coseno y BM25 están en escalas distintas). Con
        k = 60 el máximo es 2/60 ≈ 0.033 (primer lugar en las dos listas)."""
        fusion: dict[str, float] = {}
        for ranking in rankings:
            for r, d in enumerate(ranking):
                fusion[d] = fusion.get(d, 0.0) + 1.0 / (k + r)
        return fusion

    def buscar(self, consulta: str, num_docs: int = config.RAG_NUM_DOCS_TO_FETCH, tipos: list[str] | None = None,
               query_embedding: np.ndarray | None = None, consulta_lexica: str | None = None,
               rrf_k: int | None = None, modo: str = "hibrida", excluir: set[str] | None = None,
               materias: list[str] | None = None) -> list[Candidato]:
        """Búsqueda híbrida: embeddings (fragmentos agrupados por sentencia) + BM25, fusionados con RRF.

        `modo` permite las líneas base de la evaluación: "densa" (solo embeddings) o "lexica" (solo BM25); en
        ambos casos `relevancia` es 1/(k + rango) de esa única lista. `excluir` quita documentos del resultado
        (evaluación *leave-one-out*: que la sentencia fuente no vote por su propio tipo). Sin `tipos`, la búsqueda
        se restringe a `config.MATERIAS_ACTIVAS` (o a `materias`)."""
        if modo not in ("hibrida", "densa", "lexica"):
            raise ValueError(f"modo desconocido: {modo}")
        k = rrf_k or self.RRF_K
        q = embed_query(consulta) if query_embedding is None else query_embedding
        mats = None if tipos else self.materias_filtro(materias)
        densos = self._densa(q, tipos, mats) if modo != "lexica" else {}
        rank_denso = [d for d, _ in sorted(densos.items(), key=lambda kv: kv[1], reverse=True)]
        rank_lex = self.ranking_lexico(consulta_lexica or consulta, tipos, mats) if modo != "densa" else []
        if excluir:
            rank_denso = [d for d in rank_denso if d not in excluir]
            rank_lex = [d for d in rank_lex if d not in excluir]

        fusion = self.fusion_rrf([r for r in (rank_denso, rank_lex) if r], k)
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
