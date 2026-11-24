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

# --- Evaluación ---
EVAL_DIR = Path(_env("COPILOTO_EVAL_DIR", str(ROOT_DIR / "evaluacion")))
FAMILIAS_CSV = EVAL_DIR / "familias_penal.csv"            # Tipo -> familia de delito
CONSULTAS_PATH = EVAL_DIR / "consultas_penal.jsonl"         # Consultas redactadas a partir de sentencias
CONCEPTOS_CACHE_PATH = EVAL_DIR / "cache" / "conceptos.json"
RESULTADOS_DIR = EVAL_DIR / "resultados"
EVAL_SEED = _env("COPILOTO_EVAL_SEED", 42)
EVAL_BOOTSTRAP = 1000
# Modelos de Ollama que compara `evaluacion modelos` (solo los ya instalados; los demás se omiten, no se descargan).
MODELOS_COMPARACION = ["llama3.1:8b", "qwen2.5:7b", "gemma2:9b", "mistral:7b"]
# Evaluación automática: no hay abogados disponibles, así que la referencia la dan la categoría del STJ y el texto de
# la sentencia leído por reglas. Un juez LLM de otra familia que la de la app (`evaluacion jueces`, uno a la vez por
# los 8 GB de VRAM) se usa solo si acierta la familia del delito de la categoría del STJ al menos en
# JUEZ_EXACTITUD_MINIMA de las sentencias, criterio fijado antes de correrlo. Ninguno lo alcanzó: gemma2:9b 0.742
# [0.709, 0.773] en 725 sentencias del corpus final (`20261121-031724_723fe6b_referencias`; errores en
# evaluacion/errores/juez_familia.md), mistral:7b 0.795 en 1,412 (`20261117-065346_a60ca4f_referencias`) y
# qwen2.5:7b en el piloto de 60; qwen2.5:14b no cabe en la GPU. Por eso acto, apelante y sentido se leen por reglas.
MODELOS_JUECES = ["gemma2:9b"]
JUEZ_EXACTITUD_MINIMA = 0.80
# Muestra al azar de los jueces: juzgar todo tomaba ~7 h por juez. Cada elemento entra a la
# muestra por el hash de su id con la semilla (no por orden), así que al crecer el corpus los que ya estaban no
# cambian. Sentencias 0.30 (~420 de 1,412): IC95 de ±4 puntos para la exactitud de cada juez. Consultas 0.50: entran
# con todos sus pares, para que las métricas por consulta (nDCG) tengan todos sus juicios. `--completa` usa todo.
MUESTRA_SENTENCIAS = _env("COPILOTO_MUESTRA_SENTENCIAS", 0.30)
MUESTRA_CONSULTAS = _env("COPILOTO_MUESTRA_CONSULTAS", 0.50)
JUECES_DIR = EVAL_DIR / "jueces"

COLLECTION_NAME = "sentencias_judiciales"

# --- Embeddings ---
# Modelo multilingüe: el corpus está en español.
EMBEDDING_MODEL_NAME = _env("COPILOTO_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = _env("COPILOTO_EMBEDDING_BATCH_SIZE", 64)
CHUNK_SIZE_CHARS = 1500       # ~400 tokens: cabe en la ventana de 512 tokens del modelo
CHUNK_OVERLAP_CHARS = 200
MAX_CHUNKS_PER_DOC = 12       # Fragmentos repartidos uniformemente por sentencia (materias fuera del alcance)
MAX_CHUNKS_PENAL = 0          # Sentencias penales: todos los fragmentos (0 = sin tope). Con tope 12 se indexaba ~40 %

# --- Balance del corpus ---
MAX_SENTENCIAS_POR_TIPO = _env("COPILOTO_MAX_POR_TIPO", 30)

# --- Procesamiento de PDFs (Fase 1) ---
# Los PDF se leen completos (0 = sin límite); muestrearlos a 40 páginas dejaba páginas omitidas en el 57 % de las
# sentencias penales.
PDF_MAX_PAGES_FULL_READ = _env("COPILOTO_PDF_MAX_PAGINAS", 0)
PDF_TARGET_TOTAL_PAGES_SAMPLE = 40
PDF_FIXED_START_PAGES = 10
PDF_FIXED_END_PAGES = 15
MAX_WORKERS_FASE_1 = None
# OCR (Tesseract, español) de las páginas sin capa de texto, como en los PDF escaneados. "auto": se usa si
# `pdftoppm`, `tesseract` y su modelo `spa` están instalados (TESSDATA_PREFIX si el modelo está en una carpeta de
# usuario); "false": nunca. Una página con menos de PDF_OCR_MIN_CARACTERES caracteres extraídos se trata como imagen.
PDF_OCR = _env("COPILOTO_PDF_OCR", "auto")
PDF_OCR_MIN_CARACTERES = 30
PDF_OCR_DPI = 300

# --- RAG y clasificación (Fase 3) ---
RAG_NUM_RESULTS_DESIRED = _env("COPILOTO_NUM_PRECEDENTES", 3)
RAG_NUM_DOCS_TO_FETCH = 30        # Sentencias candidatas a revisar
RAG_NUM_CHUNKS_TO_FETCH = 300     # Fragmentos que se recuperan para agrupar por sentencia
RRF_K = _env("COPILOTO_RRF_K", 60)  # Constante k de Reciprocal Rank Fusion (valor propuesto por Cormack et al.)
TOP_N_CANDIDATES_INITIAL = 20
TOP_N_CANDIDATES_FINAL = 10

# Pesos del score híbrido de categorías (suman 1; solo importa el orden dentro de una consulta). KEM tiene peso 0:
# con los conceptos de la consulta como "términos importantes", KEM = |A∩B|/|A| y Jaccard = |A∩B|/|A∪B| usan los
# mismos conjuntos (correlación 0.995 en la ablación), así que su peso pasa a Jaccard. KEM sobre un vocabulario
# jurídico sin Jaccard perdió un caso en la regresión (13/14 frente a 14/14; 20261112-033711_*_regresion). La línea
# base que compara la evaluación: coseno 0.20, Jaccard 0.20, KEM 0.20, concisión 0.05, voto 0.35.
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


# --- Alcance: materia penal en apelación. Las alternativas que la evaluación usa como línea base quedan por variable
# de entorno. ---
# Penalización ×0.70 a categorías con palabras de IRRELEVANT_TOKENS ("violacion", "falsedad"...): castigaba delitos
# reales (penal/violacion tiene 30 sentencias) sin evidencia a favor. Desactivada; en la regresión solo ayudaba a un
# ejemplo por accidente, y ese caso se resuelve con ENRIQUECER_FAMILIAS
# (evaluacion/resultados/20261112-034052_79c80ac_regresion.md).
USAR_LISTA_TOKENS_PENALIZADOS = _env("COPILOTO_USAR_LISTA_TOKENS_PENALIZADOS", False)
MATERIAS_ACTIVAS = _env_lista("COPILOTO_MATERIAS", ["penal"])
NIVEL_CLASIFICACION = _env("COPILOTO_NIVEL_CLASIFICACION", "familia")  # "tipo" | "familia" (solo penal se agrupa)
# El embedding de una familia incluye los nombres de sus tipos más generales ("contra la salud: narcomenudeo..."),
# sin "y otro(s)" ni nombres compuestos: la familia correcta queda primera en 13 de 14 ejemplos penales, frente a 12
# sin ella (regresiones 20261112-180034_139bfe6 y 20261112-180217_99bb183).
ENRIQUECER_FAMILIAS = _env("COPILOTO_ENRIQUECER_FAMILIAS", True)
# QC_1 acepta mismo delito, misma familia o mismo problema procesal; exigir "mismo tema Y contexto" con el inicio del
# documento descartaba el 55 % de los 5 primeros candidatos de los ejemplos, frente al 14 % (`qc-diagnostico`,
# 20261112-184123_2c7be47). Lee la ficha precalculada de la sentencia (delito, acto impugnado, agravios, sentido); sin
# ficha vigente, el fragmento más parecido a la consulta. En 660 pares de desarrollo, por pares, leer la ficha en
# lugar del fragmento sube el recall de 0.725 a 0.844 (+0.120 [+0.036, +0.204]) sin cambio significativo en la
# especificidad (0.410 → 0.396); precisión 0.294 → 0.321 (`20261123-230824_5fc9b67_qc`).
QC1_CONTEXTO = _env("COPILOTO_QC1_CONTEXTO", "ficha")  # "inicio" | "fragmento_similar" | "ficha"
# QC_1 razona en dos frases (delito y problema de la consulta y del documento) antes del veredicto, con el modelo de la
# app. Probadas en los mismos 660 pares (`20261123-234717_e28a600_qc`) y descartadas: pedir solo el veredicto (0.59 s por
# llamada, pero responde SI en 649 de 660 pares: especificidad 0.022, deja de filtrar) y qwen2.5:7b (recall 0.844 →
# 0.425; además no cabe junto a llama3.1:8b en 8 GB y cada cambio de modelo cuesta unos 5-8 s de carga).
PROMPT_QC1_VERSION = _env("COPILOTO_PROMPT_QC1", "v11")
QC1_MAX_CHARS = 2500
# Conceptos clave (paso 1): el prompt pide primero el delito con su modalidad y después el problema procesal.
PROMPT_CONCEPTOS_VERSION = _env("COPILOTO_PROMPT_CONCEPTOS", "v11")
# Ficha penal: "json" (una llamada que devuelve la ficha estructurada; el informe se arma con código) o
# "dos_llamadas" (extracción + informe; línea base y respaldo si el JSON no es válido).
FICHA_MODO = _env("COPILOTO_FICHA_MODO", "json")
# Orden final de los candidatos: "rrf" (el de la búsqueda) o "familia_acto": si la consulta nombra un acto impugnado
# (vinculación, prisión preventiva, sentencia...), primero los de la misma familia de delito y el mismo acto, leídos
# por reglas (metadatos.py). En los 16 ejemplos penales, de los 3 primeros candidatos son de la familia correcta 32/42
# con RRF y 37/42 por facetas, sin que ningún ejemplo pierda; en prisión preventiva por homicidio pasan de
# robo, robo, homicidio a homicidio x3 (regresiones 20261113-000836_137c238, sin la condición del acto, y
# 20261113-001214_a5c476b, con ella).
ORDEN_CANDIDATOS = _env("COPILOTO_ORDEN_CANDIDATOS", "familia_acto")
# Re-ranking con cross-encoder (experimento, apagado): p. ej. "BAAI/bge-reranker-v2-m3". Se activa solo si en
# `recuperacion` la diferencia pareada en nDCG@5 con juicios humanos es positiva y su IC 95 % no incluye 0.
RERANKER = _env("COPILOTO_RERANKER", "") or None
RERANKER_DISPOSITIVO = _env("COPILOTO_RERANKER_DISPOSITIVO", "auto")  # "auto" | "cuda" | "cpu"
# Fichas precalculadas (`evaluacion fichas-batch`): la app usa la ficha guardada de cada sentencia si coinciden su
# texto y la versión (prompt + modelo); si no, la genera en la consulta. Activado por defecto solo si existe el archivo.
FICHAS_PRECALCULADAS_PATH = DATA_DIR / "fichas" / "fichas-penal.parquet"
FICHAS_PRECALCULADAS = _env("COPILOTO_FICHAS_PRECALCULADAS", FICHAS_PRECALCULADAS_PATH.exists())
# Modelo de respaldo de las fichas precalculadas: `fichas-batch --respaldo` genera con él, por lotes y sin otro
# modelo cargado, las fichas de las sentencias que no tienen una válida del modelo principal, fuera de la muestra de
# la referencia (la evaluación mide solo las del modelo principal). La app usa primero la del modelo principal y, si
# no hay, la de respaldo, en lugar de generarla durante la consulta. qwen2.5:7b: cabe en 8 GB y en `modelos`
# (20261113-125326_fb1f3fe) tardó 18 s por ficha frente a 29 s, con JSON válido 20/20. Vacío = sin respaldo.
FICHAS_MODELO_RESPALDO = _env("COPILOTO_FICHAS_MODELO_RESPALDO", "qwen2.5:7b") or None
# QC_2: "determinista" (sin LLM: el delito de la ficha debe ser de una familia clasificada o de la etiqueta de la
# sentencia; además se marcan sentido del fallo y artículos no encontrados) o "llm" (verificador LLM, línea base).
QC2_MODO = _env("COPILOTO_QC2_MODO", "determinista")
# Prompt penal: pide acto impugnado, agravios, su calificación y el sentido del fallo como valor cerrado.
PROMPT_PENAL_VERSION = _env("COPILOTO_PROMPT_PENAL", "v11")
# Síntesis (paso 5): parte de la lista por reglas de quién apeló, qué resolvió la Sala y a qué parte favoreció; no
# pronostica el resultado (una síntesis sin esas reglas leyó al revés un precedente y concluyó que "es probable que la
# apelación tenga éxito"). Se probó agregar un ejemplo de líneas citadas y una revisión final de las citas: con
# contenido jurídico en el ejemplo (v12), la primera medición mejoraba las citas (`20261123-201729_e566e3d_sintesis-citas`), pero con los
# precedentes que deja QC_1 sobre la ficha no se reprodujo (−0.024 [−0.144, +0.078]), copió el ejemplo en 9 de 19
# síntesis y pronosticó en una; con un ejemplo solo de formato (v13), +0.024 [−0.069, +0.121], no significativo
# (`20261124-003427_905c255_sintesis-citas`). Ninguna cumple el criterio y la app sigue con v11.
PROMPT_SINTESIS_VERSION = _env("COPILOTO_PROMPT_SINTESIS", "v11")
# La síntesis cita cada afirmación ([P1], [P2]...) y el código verifica que la cita exista y que el sentido del
# fallo o el delito mencionados coincidan con la ficha citada; si no, la frase se marca "verificar".
VERIFICAR_CITAS = _env("COPILOTO_VERIFICAR_CITAS", True)
# Resultados guardados para la demo (cache_resultados.py): la app muestra al instante una consulta ya analizada con
# la misma configuración. Guarda la consulta en data/cache/resultados (local). La evaluación nunca la usa.
CACHE_RESULTADOS = _env("COPILOTO_CACHE_RESULTADOS", True)
# Aviso "parece de otra materia": el corpus es penal. Se avisa si la consulta no menciona ningún delito ni término del
# proceso penal (lista en pipeline._RE_PENAL); sin él, las consultas civil, familiar y fuera de tema recibían
# precedentes penales sin aviso (`robustez`, 20261113-115133_fb1f3fe). tests/test_robustez.py comprueba que avisa en
# las 3 de otra materia y en ninguno de los ejemplos penales.
AVISO_OTRA_MATERIA = _env("COPILOTO_AVISO_OTRA_MATERIA", True)
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
    # En `modelos` (20261113-125326_fb1f3fe: 10 consultas, 20 fichas, sin referencia) qwen2.5:7b fue más rápido
    # (83 s frente a 125 s por consulta; 18 s frente a 29 s por ficha) y mejor en los diagnósticos (JSON 20/20 frente
    # a 19/20; "dato no disponible" 3.5 % frente a 8.9 %). Se mantiene llama3.1:8b porque cambiarlo invalida
    # las 2,495 fichas precalculadas (~12.7 h) y obliga a repetir la evaluación; es el siguiente cambio recomendado,
    # medido antes con `fichas-referencia` contra la referencia.
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
