"""Configuración centralizada del Co-piloto Jurídico.

Todos los valores pueden sobreescribirse con variables de entorno (ver `.env.example`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv es opcional
    pass


def _env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "si", "sí", "yes"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


# --- Rutas del proyecto ---
ROOT_DIR = Path(_env("COPILOTO_ROOT", str(Path(__file__).resolve().parents[2])))
DATA_DIR = ROOT_DIR / "data"
CATALOGO_PATH = DATA_DIR / "catalogo" / "catalogo_tipos_stj.csv"
CORPUS_DIR = DATA_DIR / "corpus"          # Texto limpio de las sentencias (versionado en git)
INDICE_DIR = DATA_DIR / "indice"          # Embeddings precalculados (versionados en git)
RAW_PDF_DIR = Path(_env("COPILOTO_PDF_DIR", str(DATA_DIR / "raw" / "pdfs")))  # PDFs descargados (no versionados)
CHROMA_DIR = Path(_env("COPILOTO_CHROMA_DIR", str(DATA_DIR / "chroma")))       # DB vectorial local (se reconstruye)
CACHE_DIR = DATA_DIR / "cache"

COLLECTION_NAME = "sentencias_judiciales"
MATERIAS = ["penal", "civil", "familiar", "mercantil", "laboral"]

# --- Embeddings ---
# Modelo multilingüe (el corpus está en español). all-mpnet-base-v2 era un modelo entrenado en inglés.
EMBEDDING_MODEL_NAME = _env("COPILOTO_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = _env("COPILOTO_EMBEDDING_BATCH_SIZE", 64)
CHUNK_SIZE_CHARS = 1500       # ~400 tokens: cabe en la ventana de 512 tokens del modelo
CHUNK_OVERLAP_CHARS = 200
MAX_CHUNKS_PER_DOC = 12       # Fragmentos repartidos uniformemente por sentencia (controla el tamaño del índice)

# --- Balance del corpus ---
MAX_SENTENCIAS_POR_TIPO = _env("COPILOTO_MAX_POR_TIPO", 30)

# --- Descarga (API pública del STJ Jalisco) ---
API_BASE_URL = "https://publica-sentencias-backend.stjjalisco.gob.mx"
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 4

# --- Procesamiento de PDFs (Fase 1) ---
PDF_MAX_PAGES_FULL_READ = 50
PDF_TARGET_TOTAL_PAGES_SAMPLE = 40
PDF_FIXED_START_PAGES = 10
PDF_FIXED_END_PAGES = 15
MAX_WORKERS_FASE_1 = None

# --- RAG y clasificación (Fase 3) ---
RAG_NUM_RESULTS_DESIRED = _env("COPILOTO_NUM_PRECEDENTES", 3)
RAG_NUM_DOCS_TO_FETCH = 30        # Sentencias candidatas a revisar
RAG_NUM_CHUNKS_TO_FETCH = 300     # Fragmentos que se recuperan para agrupar por sentencia
TOP_N_CANDIDATES_INITIAL = 20
TOP_N_CANDIDATES_FINAL = 10

# Pesos para Re-ranking Híbrido (v10: se añade el voto de recuperación y se reducen Jaccard/KEM, que con
# 567 tipos finos sobrevaloraban coincidencias de una sola palabra, p. ej. "menores")
COSINE_WEIGHT = 0.20
JACCARD_WEIGHT = 0.20
KEM_WEIGHT = 0.20
CONCISION_WEIGHT = 0.05
VOTE_WEIGHT = 0.35
NUM_CATEGORIAS_FILTRO = 3         # El filtro estricto usa las 3 mejores categorías
IRRELEVANT_TOKENS_PENALTY_FACTOR = 0.70
KEM_THRESHOLD_FOR_PENALTY = 0.6
MAX_TOKENS_FOR_PENALTY = 5
IRRELEVANT_TOKENS = {"desmantelamiento", "falsedad", "informe", "violacion", "medica", "medico"}

FAILURE_KEYWORDS = [
    "lo siento", "no puedo", "plantilla general", "sin hechos", "hechos extraídos que no",
    "hipotética", "asistencia legal", "no proporcionar", "información específica ni contexto",
    "no tengo información", "no puedo analizar",
]


@dataclass
class LLMSettings:
    """Parámetros del proveedor de LLM. Se editan desde la barra lateral de la app."""

    provider: str = _env("COPILOTO_LLM_PROVIDER", "ollama")  # "ollama" | "openai"
    model: str = _env("COPILOTO_LLM_MODEL", "llama3.1:8b")
    base_url: str = _env("COPILOTO_LLM_BASE_URL", "http://localhost:11434")
    api_key: str = field(default=_env("COPILOTO_LLM_API_KEY", ""), repr=False)
    temperature: float = _env("COPILOTO_LLM_TEMPERATURE", 0.0)
    max_tokens: int = _env("COPILOTO_LLM_MAX_TOKENS", 1024)
    # Ventana de contexto de Ollama. Sin fijarla, Ollama recorta silenciosamente a 2-4k tokens.
    num_ctx: int = _env("COPILOTO_LLM_NUM_CTX", 8192)
    timeout: int = _env("COPILOTO_LLM_TIMEOUT", 600)

    @property
    def max_context_chars(self) -> int:
        """Caracteres de sentencia que caben en el prompt (~3.5 caracteres por token en español)."""
        reservados = self.max_tokens + 900  # salida + plantilla del prompt
        return max(4000, int((self.num_ctx - reservados) * 3.2))
