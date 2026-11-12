"""Página "Cómo funciona": flujo, límites de tamaño, glosario jurídico y uso responsable."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.ayudas import AYUDA, GLOSARIO, USO_RESPONSABLE
from copiloto_juridico.pipeline import DESCRIPCION_PASOS, PASOS

st.set_page_config(page_title="Cómo funciona · Co-piloto Jurídico", page_icon="🧭", layout="wide")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_common import inject_css  # noqa: E402

inject_css()
llm = config.LLMSettings()

st.title("🧭 Cómo funciona")
st.markdown(
    "El Co-piloto ayuda a un **defensor penal** que prepara o contesta una **apelación ante las Salas Penales del "
    "STJ Jalisco** a encontrar cómo ha resuelto el propio tribunal casos parecidos. Recibe la descripción del caso "
    "y devuelve las sentencias más parecidas, una ficha de cada una y una comparación entre ellas.")

st.header("1. El flujo, paso a paso")
st.markdown("**0. Ingesta e indexación (una sola vez):** PDF → texto limpio → fragmentos → embeddings + índice BM25.")
for n, (clave, titulo) in enumerate(PASOS, 1):
    st.markdown(f"**{n}. {titulo}.** {DESCRIPCION_PASOS[clave]}")
st.info(f"Con {config.RAG_NUM_RESULTS_DESIRED} precedentes y ningún descarte se hacen 1 + "
        f"{config.RAG_NUM_RESULTS_DESIRED}×4 + 1 = {2 + 4 * config.RAG_NUM_RESULTS_DESIRED} llamadas al LLM; cada "
        "descarte añade llamadas. Por eso una consulta tarda minutos en una GPU de consumo.", icon="⏱️")

st.header("2. Los tres límites de tamaño (no confundirlos)")
st.markdown("Son tres mecanismos distintos, en tres etapas distintas:")
reservados = llm.max_tokens + 900
st.table([
    {"Límite": "Ventana del modelo de embeddings", "Valor": "512 tokens",
     "Qué pasa si se excede": "El modelo trunca y solo representa el inicio del texto.",
     "Cómo lo maneja el sistema": f"La sentencia se divide en fragmentos de {config.CHUNK_SIZE_CHARS:,} caracteres "
                                  f"(≈ 400 tokens) con {config.CHUNK_OVERLAP_CHARS} de traslape; hasta "
                                  f"{config.MAX_CHUNKS_PER_DOC} por sentencia, siempre incluyendo inicio y final.",
     "Código": "embeddings.chunk_text"},
    {"Límite": "Ventana de contexto del LLM", "Valor": f"{llm.num_ctx:,} tokens (prompt + respuesta)",
     "Qué pasa si se excede": "Ollama descarta texto sin avisar.",
     "Cómo lo maneja el sistema": f"Se reservan {reservados:,} tokens para plantilla y respuesta; caben ≈ "
                                  f"{llm.max_context_chars:,} caracteres de sentencia. Si no cabe, se conserva el 55 % "
                                  "inicial y el 45 % final (hechos y resolutivos). **No se resume: se recorta.**",
     "Código": "pipeline.recortar_contexto"},
    {"Límite": "PDFs largos (ingesta)", "Valor": f"{config.PDF_MAX_PAGES_FULL_READ} páginas",
     "Qué pasa si se excede": "Tiempo de extracción y texto excesivo.",
     "Cómo lo maneja el sistema": f"Se leen {config.PDF_TARGET_TOTAL_PAGES_SAMPLE} páginas: "
                                  f"{config.PDF_FIXED_START_PAGES} iniciales, {config.PDF_FIXED_END_PAGES} finales y el "
                                  "resto intermedias equiespaciadas.",
     "Código": "datos/procesamiento.pages_to_read"},
])
st.caption("Un token es la unidad en que un modelo divide el texto (palabra, parte de palabra o signo); en español "
           "equivale a unos 3.2–4 caracteres. El código usa 3.2 como estimación conservadora. Consecuencia: en una "
           "sentencia mediana (≈ 50,000 caracteres) el LLM lee alrededor del 40 % del texto al extraer hechos.")

st.header("3. Cómo leer los números")
c1, c2 = st.columns(2)
with c1:
    for k, nombre in (("similitud", "Similitud"), ("rrf", "RRF"), ("score_hibrido", "Score híbrido"),
                      ("descartados", "Descartados por QC"), ("ambiguos", "Veredictos ambiguos")):
        st.markdown(f"**{nombre}.** {AYUDA[k]}")
with c2:
    for k, nombre in (("coseno", "Coseno de categoría"), ("jaccard", "Jaccard"), ("kem", "KEM"),
                      ("concision", "Concisión"), ("voto", "Voto de recuperación"), ("penalidad", "Penalidad")):
        st.markdown(f"**{nombre}.** {AYUDA[k]}")

st.header("4. Glosario jurídico")
st.caption("Definiciones operativas para este proyecto; se validan con el asesor jurídico.")
for termino, definicion in GLOSARIO.items():
    st.markdown(f"**{termino}.** {definicion}")

st.header("5. Uso responsable")
st.warning(USO_RESPONSABLE, icon="⚖️")
st.markdown(
    "- **Cobertura:** muchos tipos penales tienen una o ninguna sentencia; cuando ningún precedente sale de las "
    "categorías detectadas, la app lo dice explícitamente.\n"
    "- **Lectura parcial:** el LLM lee una parte de cada sentencia y el índice representa alrededor de un tercio; "
    "puede omitir datos del centro del documento.\n"
    "- **Etiquetas del portal:** la categoría de cada sentencia proviene del nombre de archivo que asigna el STJ.\n"
    "- **Sin OCR:** los PDFs escaneados no aportan texto.\n"
    "- **Datos testados:** los datos personales eliminados por el STJ se reportan como no disponibles; nunca se infieren.")
