"""Página "Evaluación": últimos resultados medidos, con IC 95 % y la explicación de cada métrica.

No muestra ningún número que no esté en `evaluacion/resultados/`: si no hay corridas, dice "por medir".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.ayudas import METRICAS
from copiloto_juridico.evaluacion.reporte import ultimo_resultado

st.set_page_config(page_title="Evaluación · Co-piloto Jurídico", page_icon="📏", layout="wide")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_common import inject_css  # noqa: E402

inject_css()

COMANDOS = [
    ("recuperacion", "Recuperación y clasificación",
     "¿El filtro conserva la categoría correcta? ¿La búsqueda híbrida supera a BM25 y a embeddings por separado?"),
    ("ablacion", "Ablación del clasificador", "¿Qué componente del score híbrido aporta y cuál sobra?"),
    ("pesos", "Ajuste de pesos", "¿Los pesos manuales de v10 están cerca del óptimo? (se eligen en desarrollo)"),
    ("qc", "Controles de calidad", "¿QC_1 y QC_2 coinciden con el juicio de un abogado? ¿Cuántos precedentes útiles pierden?"),
    ("generacion", "Calidad de la ficha", "¿La ficha reporta bien el sentido del fallo, la etapa, el delito y los artículos?"),
    ("latencia", "Eficiencia", "¿Cuánto tarda una consulta y cuántas llamadas al LLM hace?"),
]

st.title("📏 Evaluación")
st.markdown(
    "Resultados del protocolo de evaluación (sección 6 del informe). Cada tabla viene de una corrida guardada en "
    "`evaluacion/resultados/` con su fecha, commit, configuración y huella del corpus. **Los valores que no se han "
    "medido se muestran como _por medir_.**")

with st.expander("📖 Qué significa cada métrica", expanded=False):
    for nombre, definicion in METRICAS.items():
        st.markdown(f"**{nombre}.** {definicion}")

st.info("Referencia histórica (v10): en 15 de 18 consultas de ejemplo el filtro incluyó la categoría correcta, "
        "Hit@3 = 0.83, IC 95 % [0.61, 0.94], n = 18. Son consultas de cinco materias, no un conjunto penal, y los "
        "pesos se ajustaron mirando esas mismas consultas: es una estimación optimista.", icon="ℹ️")


def mostrar(datos: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Fecha", datos["fecha"].replace("T", " ")[:16])
    cols[1].metric("Commit", datos["commit"] + (" *" if datos.get("sucio") else ""),
                   help="* = había cambios sin confirmar en src/ o app/ al correr.")
    cols[2].metric("Consultas", datos.get("n_consultas", "—"))
    cols[3].metric("Partición", (datos.get("parametros") or {}).get("particion") or "—",
                   help="Desarrollo = para elegir configuración; prueba = el número que se reporta (una sola vez).")
    if datos.get("borrador"):
        st.error("**BORRADOR — no reportable:** la corrida usó consultas marcadas como borrador.", icon="🚧")
    for a in datos.get("advertencias", []):
        st.warning(a, icon="⚠️")
    st.caption(f"Archivo: `{datos['_archivo']}` · corpus `{datos.get('corpus_fingerprint')}` · "
               f"embeddings `{datos.get('modelo_embeddings')}` · LLM `{datos.get('modelo_llm')}` · "
               f"semilla {datos.get('semilla')}")
    for nombre, tabla in datos.get("tablas", {}).items():
        st.markdown(f"#### {nombre}")
        st.caption(tabla.get("descripcion", ""))
        df = pd.DataFrame(tabla["filas"], columns=tabla["columnas"])
        cfg = {}
        for c in df.columns:
            ayuda = next((v for k, v in METRICAS.items() if c.startswith(k) or k in c), None)
            if pd.api.types.is_float_dtype(df[c]):
                cfg[c] = st.column_config.NumberColumn(format="%.3f", help=ayuda)
            elif ayuda:
                cfg[c] = st.column_config.Column(help=ayuda)
        st.dataframe(df, width="stretch", hide_index=True, column_config=cfg)


for comando, titulo, pregunta in COMANDOS:
    st.header(titulo)
    st.markdown(f"_{pregunta}_")
    datos = ultimo_resultado(comando)
    if datos is None:
        st.markdown("**Por medir.** Para obtenerlo:")
        st.code(f"python -m copiloto_juridico.evaluacion {comando}"
                + (" --particion desarrollo" if comando in ("recuperacion", "ablacion", "qc", "latencia") else ""),
                language="bash")
        continue
    mostrar(datos)

st.divider()
st.caption(f"Guía de anotación y orden de trabajo: `{config.EVAL_DIR.name}/README.md`.")
