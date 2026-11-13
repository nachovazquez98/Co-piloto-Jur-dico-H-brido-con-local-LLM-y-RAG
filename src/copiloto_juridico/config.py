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

# --- Evaluación (v11) ---
EVAL_DIR = Path(_env("COPILOTO_EVAL_DIR", str(ROOT_DIR / "evaluacion")))
FAMILIAS_CSV = EVAL_DIR / "familias_penal.csv"            # Tipo -> familia de delito (lo revisa un abogado)
CONSULTAS_PATH = EVAL_DIR / "consultas_penal.jsonl"         # Consultas derivadas de sentencias (borradores revisados)
CONSULTAS_REALES_PATH = EVAL_DIR / "consultas_reales.jsonl"  # Escritas por abogados sin ver sentencias
PILOTO_JUICIOS_PATH = EVAL_DIR / "piloto_juicios.csv"       # Piloto de anotación: 10 consultas x 5, dos anotadores
JUICIOS_PATH = EVAL_DIR / "anotacion_juicios.csv"           # Hoja de juicios de relevancia 0/1/2 (la llena el abogado)
FICHAS_ANOTADAS_PATH = EVAL_DIR / "anotacion_fichas.csv"    # Hoja del conjunto C (la llena el abogado)
CONCEPTOS_CACHE_PATH = EVAL_DIR / "cache" / "conceptos.json"
RESULTADOS_DIR = EVAL_DIR / "resultados"
EVAL_SEED = _env("COPILOTO_EVAL_SEED", 42)
EVAL_BOOTSTRAP = 1000

COLLECTION_NAME = "sentencias_judiciales"
MATERIAS = ["penal", "civil", "familiar", "mercantil", "laboral"]

# --- Embeddings ---
# Modelo multilingüe (el corpus está en español). all-mpnet-base-v2 era un modelo entrenado en inglés.
EMBEDDING_MODEL_NAME = _env("COPILOTO_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = _env("COPILOTO_EMBEDDING_BATCH_SIZE", 64)
CHUNK_SIZE_CHARS = 1500       # ~400 tokens: cabe en la ventana de 512 tokens del modelo
CHUNK_OVERLAP_CHARS = 200
MAX_CHUNKS_PER_DOC = 12       # Fragmentos repartidos uniformemente por sentencia (materias fuera del alcance)
MAX_CHUNKS_PENAL = 0          # Sentencias penales: todos los fragmentos (0 = sin tope). Con tope 12 se indexaba ~40 %

# --- Balance del corpus ---
MAX_SENTENCIAS_POR_TIPO = _env("COPILOTO_MAX_POR_TIPO", 30)

# --- Descarga (API pública del STJ Jalisco) ---
API_BASE_URL = "https://publica-sentencias-backend.stjjalisco.gob.mx"
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 4

# --- Procesamiento de PDFs (Fase 1) ---
# v11: los PDF se leen completos (0 = sin límite). Antes, los de más de 50 páginas se muestreaban a 40 y se perdía
# contexto: el 57 % de las sentencias penales del corpus actual tiene páginas omitidas.
PDF_MAX_PAGES_FULL_READ = _env("COPILOTO_PDF_MAX_PAGINAS", 0)
PDF_TARGET_TOTAL_PAGES_SAMPLE = 40
PDF_FIXED_START_PAGES = 10
PDF_FIXED_END_PAGES = 15
MAX_WORKERS_FASE_1 = None

# --- RAG y clasificación (Fase 3) ---
RAG_NUM_RESULTS_DESIRED = _env("COPILOTO_NUM_PRECEDENTES", 3)
RAG_NUM_DOCS_TO_FETCH = 30        # Sentencias candidatas a revisar
RAG_NUM_CHUNKS_TO_FETCH = 300     # Fragmentos que se recuperan para agrupar por sentencia
RRF_K = _env("COPILOTO_RRF_K", 60)  # Constante k de Reciprocal Rank Fusion (valor propuesto por Cormack et al.)
TOP_N_CANDIDATES_INITIAL = 20
TOP_N_CANDIDATES_FINAL = 10

# Pesos del score híbrido de categorías (suman 1; solo importa el orden dentro de una consulta).
# v11: KEM sale del score (peso 0) y su peso pasa a Jaccard. Con los conceptos de la consulta como "términos
# importantes", KEM = |A∩B|/|A| y Jaccard = |A∩B|/|A∪B| usan los mismos conjuntos (correlación 0.995 en la
# ablación). La alternativa, KEM sobre un vocabulario jurídico sin Jaccard, perdió un caso en la regresión de las
# 18 consultas v10 (13/14 frente a 14/14): evaluacion/resultados/20261112-033711_*_regresion.md.
# Pesos de v10: coseno 0.20, Jaccard 0.20, KEM 0.20, concisión 0.05, voto 0.35.
COSINE_WEIGHT = 0.20
JACCARD_WEIGHT = 0.40
KEM_WEIGHT = 0.00
CONCISION_WEIGHT = 0.05
VOTE_WEIGHT = 0.35
NUM_CATEGORIAS_FILTRO = 3         # El filtro estricto usa las 3 mejores categorías

IRRELEVANT_TOKENS_PENALTY_FACTOR = 0.70
KEM_THRESHOLD_FOR_PENALTY = 0.6
MAX_TOKENS_FOR_PENALTY = 5
IRRELEVANT_TOKENS = {"desmantelamiento", "falsedad", "informe", "violacion", "medica", "medico"}


def _env_opcional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def _env_lista(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    return [v.strip() for v in value.split(",") if v.strip()] if value else list(default)


# --- v11: el alcance es la defensa penal en apelación. Las opciones de v10 quedan por variable de entorno solo
# para comparar en la evaluación (p. ej. COPILOTO_MATERIAS=penal,civil,familiar,mercantil,laboral). ---
# Penalización ×0.70 a categorías con palabras de IRRELEVANT_TOKENS ("violacion", "falsedad"...): castigaba delitos
# reales (penal/violacion tiene 30 sentencias) sin evidencia a favor. Desactivada; en la regresión solo ayudaba a un
# ejemplo por accidente, y ese caso se resuelve con ENRIQUECER_FAMILIAS
# (evaluacion/resultados/20261112-034052_79c80ac_regresion.md).
USAR_LISTA_TOKENS_PENALIZADOS = _env("COPILOTO_USAR_LISTA_TOKENS_PENALIZADOS", False)
MATERIAS_ACTIVAS = _env_lista("COPILOTO_MATERIAS", ["penal"])
NIVEL_CLASIFICACION = _env("COPILOTO_NIVEL_CLASIFICACION", "familia")  # "tipo" | "familia" (solo penal se agrupa)
# El embedding de una familia incluye los nombres de sus tipos más generales ("contra la salud: narcomenudeo..."),
# sin "y otro(s)" ni nombres compuestos. Con esa limpieza, la familia correcta queda primera en 13 de 14 ejemplos
# penales (antes 12): regresiones 20261112-180034_139bfe6 (antes) y 20261112-180217_99bb183 (después).
ENRIQUECER_FAMILIAS = _env("COPILOTO_ENRIQUECER_FAMILIAS", True)
# QC_1. En una consulta real de defensa, la versión v10 (inicio del documento + "mismo tema Y contexto") descartó
# 14 de 15 candidatos. v11: lee la ventana alrededor del fragmento más parecido a la consulta y acepta mismo delito,
# misma familia o mismo problema procesal. Sin anotaciones (`qc-diagnostico`, corrida 20261112-184123_2c7be47): en
# los 5 primeros candidatos de los 16 ejemplos penales, v10 descarta 55 % y v11 + fragmento 14 %, ninguna consulta
# se queda sin candidatos. Si esos descartes son correctos lo mide `qc` contra juicios humanos.
QC1_CONTEXTO = _env("COPILOTO_QC1_CONTEXTO", "fragmento_similar")  # "inicio" | "fragmento_similar"
PROMPT_QC1_VERSION = _env("COPILOTO_PROMPT_QC1", "v11")
QC1_MAX_CHARS = 2500
# Conceptos clave (paso 1): "v10" o "v11" (pide primero el delito y después el problema procesal).
PROMPT_CONCEPTOS_VERSION = _env("COPILOTO_PROMPT_CONCEPTOS", "v11")
# Ficha penal: "json" (v11: una llamada que devuelve la ficha estructurada; el informe se arma con código) o
# "dos_llamadas" (extracción + informe, como v10). Con "json" se ahorra una llamada por candidato.
FICHA_MODO = _env("COPILOTO_FICHA_MODO", "json")
# Orden final de los candidatos: "rrf" (el de la búsqueda) o "familia_acto": si la consulta nombra un acto impugnado
# (vinculación, prisión preventiva, sentencia...), primero los de la misma familia de delito y el mismo acto, leídos
# por reglas (metadatos.py). En los 16 ejemplos penales, de los 3 primeros candidatos son de la familia correcta 32/42
# con RRF y 37/42 por facetas, sin que ningún ejemplo pierda; en prisión preventiva por homicidio pasan de
# robo, robo, homicidio a homicidio x3 (regresiones 20261113-000836_137c238, sin la condición del acto, y
# 20261113-001214_a5c476b, con ella).
ORDEN_CANDIDATOS = _env("COPILOTO_ORDEN_CANDIDATOS", "familia_acto")
# Fichas precalculadas (`evaluacion fichas-batch`): la app usa la ficha guardada de cada sentencia si coinciden su
# texto y la versión (prompt + modelo); si no, la genera en la consulta. Activado por defecto solo si existe el archivo.
FICHAS_PRECALCULADAS_PATH = DATA_DIR / "fichas" / "fichas-penal.parquet"
FICHAS_PRECALCULADAS = _env("COPILOTO_FICHAS_PRECALCULADAS", FICHAS_PRECALCULADAS_PATH.exists())
# QC_2: "determinista" (sin LLM: el delito de la ficha debe ser de una familia clasificada o de la etiqueta de la
# sentencia; además se marcan sentido del fallo y artículos no encontrados) o "llm" (verificador LLM de v10).
QC2_MODO = _env("COPILOTO_QC2_MODO", "determinista")
# Prompts penales: "v10" o "v11" (añade acto impugnado, agravios, su calificación y sentido del fallo cerrado).
PROMPT_PENAL_VERSION = _env("COPILOTO_PROMPT_PENAL", "v11")
# Síntesis (paso 5). En una consulta real, la v10 leyó al revés un precedente que confirmó la vinculación pese a
# impugnarse el reconocimiento por fotografía y concluyó que "es probable que la apelación tenga éxito". v11: dice
# quién apeló y qué resolvió la Sala, separa precedentes a favor y en contra del argumento, y no predice resultados.
PROMPT_SINTESIS_VERSION = _env("COPILOTO_PROMPT_SINTESIS", "v11")
# Aviso "no hay precedentes del mismo tipo": siempre que ningún precedente venga del filtro; además, si se fija
# un umbral (medido con `evaluacion recuperacion`), cuando la señal de coincidencia de la mejor categoría
# queda por debajo. Sin umbral (por defecto) esa segunda condición está desactivada: "por medir".
UMBRAL_SIN_PRECEDENTES = _env_opcional_float("COPILOTO_UMBRAL_SIN_PRECEDENTES")
SENAL_SIN_PRECEDENTES = _env("COPILOTO_SENAL_SIN_PRECEDENTES", "kem_max")  # score_hibrido | coseno | kem_max

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
    # Semilla del muestreo. Con temperatura 0 reduce (no elimina del todo, en GPU) la variación entre corridas.
    seed: int = _env("COPILOTO_LLM_SEED", 42)

    @property
    def max_context_chars(self) -> int:
        """Caracteres de sentencia que caben en el prompt: (num_ctx − reservados) × 3.2, con 3.2 caracteres por
        token como estimación conservadora para español (en la práctica, 3.2–4)."""
        reservados = self.max_tokens + 900  # salida + plantilla del prompt
        return max(4000, int((self.num_ctx - reservados) * 3.2))
