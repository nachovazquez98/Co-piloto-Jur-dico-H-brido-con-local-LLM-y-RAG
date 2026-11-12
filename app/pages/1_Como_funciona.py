"""Página "Cómo funciona": el flujo, los límites de tamaño, el glosario y la última evaluación guardada."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.ayudas import AVISO_LEGAL, GLOSARIO
from copiloto_juridico.evaluacion.reporte import ultimo_resultado
from copiloto_juridico.pipeline import DESCRIPCION_PASOS, PASOS

st.set_page_config(page_title="Cómo funciona · Co-piloto Jurídico", page_icon="⚖️", layout="wide")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_common import inject_css  # noqa: E402

inject_css()
llm = config.LLMSettings()

st.title("Cómo funciona")
st.markdown("Para la defensa penal que prepara o contesta una apelación ante las Salas Penales del STJ Jalisco. "
            "Busca en sentencias públicas del propio tribunal y arma una ficha de cada precedente.")

st.header("El flujo")
for n, (clave, titulo) in enumerate(PASOS, 1):
    st.markdown(f"**{n}. {titulo}.** {DESCRIPCION_PASOS[clave]}")

st.header("Los tres límites de tamaño")
st.table(pd.DataFrame([
    {"Límite": "Modelo de embeddings", "Valor": "512 tokens",
     "Cómo se maneja": f"Cada sentencia se divide en fragmentos de {config.CHUNK_SIZE_CHARS:,} caracteres (≈ 400 "
                       "tokens); las sentencias penales se indexan completas."},
    {"Límite": "Ventana del LLM", "Valor": f"{llm.num_ctx:,} tokens",
     "Cómo se maneja": f"Caben ≈ {llm.max_context_chars:,} caracteres. Si la sentencia no cabe se recorta, no se "
                       "resume: inicio, la parte donde la Sala analiza los agravios y los resolutivos."},
    {"Límite": "PDF", "Valor": "sin límite",
     "Cómo se maneja": "Se leen todas las páginas. Parte del corpus actual se procesó antes con muestreo de "
                       "páginas; esas sentencias lo indican en el texto ('páginas omitidas')."},
]).set_index("Límite"))

st.header("Glosario")
for termino, definicion in GLOSARIO.items():
    st.markdown(f"**{termino}.** {definicion}")

st.header("Evaluación")
st.markdown("Resultados de la última corrida guardada de cada experimento (`evaluacion/resultados/`). Lo que no "
            "se ha medido aparece como **por medir**.")
for comando, titulo in (("recuperacion", "Clasificación y recuperación"), ("qc", "Controles de calidad"),
                        ("generacion", "Calidad de la ficha"), ("latencia", "Tiempo por consulta")):
    datos = ultimo_resultado(comando)
    st.subheader(titulo)
    if datos is None:
        st.markdown(f"Por medir (`python -m copiloto_juridico.evaluacion {comando}`).")
        continue
    st.caption(f"{datos['fecha'][:10]} · commit {datos['commit']} · {datos.get('n_consultas', '—')} consultas"
               + (" · BORRADOR: no reportable" if datos.get("borrador") else ""))
    for nombre, tabla in datos.get("tablas", {}).items():
        if not tabla.get("diagnostico"):
            st.markdown(f"**{nombre}**")
            st.dataframe(pd.DataFrame(tabla["filas"], columns=tabla["columnas"]), hide_index=True, width="stretch")

st.divider()
st.caption(AVISO_LEGAL)
