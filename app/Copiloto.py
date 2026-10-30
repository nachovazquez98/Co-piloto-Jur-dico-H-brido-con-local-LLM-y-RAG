"""Co-piloto Jurídico Híbrido — interfaz web (Streamlit).

Ejecutar desde la raíz del repositorio:
    streamlit run app/Copiloto.py
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime

import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.ejemplos import EJEMPLOS
from copiloto_juridico.llm import get_client
from copiloto_juridico.pipeline import PASOS, Copiloto, Opciones, Resultado

st.set_page_config(page_title="Co-piloto Jurídico", page_icon="⚖️", layout="wide")

from ui_common import get_clasificador, get_store, inject_css, sidebar_llm  # noqa: E402

inject_css()

# --- Barra lateral ---
settings = sidebar_llm()
st.sidebar.header("🔎 Análisis")
num_precedentes = st.sidebar.slider("Precedentes a analizar", 1, 5, config.RAG_NUM_RESULTS_DESIRED,
                                    help="Informes válidos que se buscan antes de la síntesis.")
qc1 = st.sidebar.toggle("QC_1 · Verificar relevancia del documento", value=True)
qc2 = st.sidebar.toggle("QC_2 · Verificar el informe (anti-alucinación)", value=True)

store = get_store()
clasificador = get_clasificador()
st.sidebar.caption(f"📚 {len(store.corpus):,} sentencias · {store.corpus['tipo'].nunique()} tipos · "
                   f"{store.manifest['num_chunks']:,} fragmentos indexados")

# --- Encabezado ---
st.markdown('<div class="cj-hero"><h1>⚖️ Co-piloto Jurídico Híbrido</h1>'
            '<p>Análisis de precedentes del Supremo Tribunal de Justicia de Jalisco con RAG y LLMs locales</p></div>',
            unsafe_allow_html=True)

# --- Consulta ---
OPCION_PROPIA = "✍️ Escribir mi propia consulta"
etiquetas = [OPCION_PROPIA] + [e.etiqueta for e in EJEMPLOS]
por_etiqueta = {e.etiqueta: e for e in EJEMPLOS}


def _cargar_ejemplo() -> None:
    ej = por_etiqueta.get(st.session_state["ejemplo"])
    st.session_state["consulta"] = ej.texto if ej else ""


if "consulta" not in st.session_state:
    st.session_state["consulta"] = EJEMPLOS[0].texto
    st.session_state["ejemplo"] = EJEMPLOS[0].etiqueta

col_a, col_b = st.columns([2, 1])
with col_a:
    st.selectbox("Consultas de ejemplo (materia · tipo de delito o acción)", etiquetas, key="ejemplo",
                 on_change=_cargar_ejemplo,
                 help="Elige un caso de ejemplo para cargarlo en el cuadro de texto; puedes editarlo antes de analizar.")
with col_b:
    ej = por_etiqueta.get(st.session_state["ejemplo"])
    if ej:
        st.markdown(f"<br><span class='cj-chip'>Materia: {ej.materia}</span>"
                    f"<span class='cj-chip'>Tipo: {html.escape(ej.tipo)}</span>", unsafe_allow_html=True)

consulta = st.text_area("Describe el caso", key="consulta", height=130,
                        placeholder="Describe los hechos, el delito o la acción y los argumentos de las partes...")
analizar = st.button("Analizar precedentes", type="primary", disabled=not consulta.strip())


# --- Ejecución con pasos animados y logs en vivo ---
def render_pasos(estado: dict) -> str:
    filas = []
    for clave, titulo in PASOS:
        e = estado[clave]
        tiempo = f"<span class='time'>{e['t']:.1f} s</span>" if e.get("t") is not None else ""
        detalle = f"<div class='detail'>{html.escape(e['detalle'])}</div>" if e.get("detalle") else ""
        filas.append(f"<div class='cj-step {e['estado']}'><span class='dot'></span>"
                     f"<div class='title'>{titulo}{tiempo}</div>{detalle}</div>")
    return "<div class='cj-steps'>" + "".join(filas) + "</div>"


def ejecutar(consulta: str) -> None:
    copiloto = Copiloto(store, clasificador, get_client(settings))
    opciones = Opciones(num_precedentes=num_precedentes, qc_relevancia=qc1, qc_post_reporte=qc2)
    estado = {k: {"estado": "pending", "detalle": "", "t": None} for k, _ in PASOS}
    inicio_paso: dict[str, float] = {}

    st.subheader("Progreso del análisis")
    col_pasos, col_logs = st.columns([1, 1.25])
    with col_pasos:
        ph_pasos = st.empty()
        ph_barra = st.empty()
    with col_logs:
        st.caption("Logs de ejecución")
        with st.container(height=340):
            ph_logs = st.empty()
    ph_vivo = st.container()
    ph_sintesis = None
    logs: list[str] = []
    sintesis = ""

    ph_pasos.markdown(render_pasos(estado), unsafe_allow_html=True)
    for ev in copiloto.analizar(consulta, opciones):
        if ev.tipo == "paso_inicio":
            inicio_paso[ev.paso] = time.time()
            estado[ev.paso]["estado"] = "running"
            if ev.paso == "sintesis":
                with ph_vivo:
                    st.markdown("#### ✍️ Redactando la síntesis...")
                    ph_sintesis = st.empty()
        elif ev.tipo == "paso_fin":
            estado[ev.paso].update(estado="done", detalle=ev.mensaje or estado[ev.paso]["detalle"],
                                   t=time.time() - inicio_paso[ev.paso])
            if ev.paso == "analisis":
                ph_barra.empty()
        elif ev.tipo == "progreso":
            estado[ev.paso]["detalle"] = ev.mensaje
            ph_barra.progress(ev.datos["validos"] / ev.datos["meta"],
                              text=f"Informes válidos: {ev.datos['validos']}/{ev.datos['meta']}")
        elif ev.tipo == "log":
            logs.append(ev.mensaje)
            ph_logs.code("\n".join(logs[-400:]), language=None)
            continue
        elif ev.tipo == "precedente":
            p = ev.datos["precedente"]
            with ph_vivo:
                st.toast(f"Precedente {p.numero} listo: {p.candidato.doc_id}", icon="📜")
        elif ev.tipo == "token" and ph_sintesis is not None:
            sintesis += ev.mensaje
            ph_sintesis.markdown(sintesis + " ▌")
            continue
        elif ev.tipo == "error":
            if ev.paso:
                estado[ev.paso].update(estado="error", detalle=ev.mensaje)
            logs.append(f"❌ {ev.mensaje}")
            ph_logs.code("\n".join(logs[-400:]), language=None)
            st.error(ev.mensaje)
        elif ev.tipo == "resultado":
            st.session_state["resultado"] = ev.datos["resultado"]
            st.session_state["logs"] = logs
            st.session_state["pasos"] = estado
        ph_pasos.markdown(render_pasos(estado), unsafe_allow_html=True)


if analizar:
    st.session_state.pop("resultado", None)
    ejecutar(consulta.strip())
    if "resultado" in st.session_state:
        st.rerun()

# --- Resultado ---
res: Resultado | None = st.session_state.get("resultado")
if res:
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precedentes válidos", len(res.precedentes))
    m2.metric("Descartados por QC", len(res.descartados))
    m3.metric("Tiempo total", f"{res.segundos:.0f} s")
    m4.metric("Modelo", res.modelo)
    st.markdown(f"<span class='cj-chip'>Categoría: {html.escape(res.tabla_clasificacion.iloc[0]['categoría'])}</span>"
                f"<span class='cj-chip'>{html.escape(res.modo_busqueda)}</span>"
                f"<span class='cj-chip'>Conceptos: {html.escape(res.entidades[:120])}</span>",
                unsafe_allow_html=True)

    if st.session_state.get("pasos"):
        with st.expander("Pasos del análisis", expanded=False):
            st.markdown(render_pasos(st.session_state["pasos"]), unsafe_allow_html=True)

    tab_s, tab_p, tab_c, tab_l = st.tabs(["📋 Síntesis", "📚 Precedentes", "🧭 Clasificación", "📝 Logs"])
    with tab_s:
        with st.container(border=True):
            st.markdown("## Síntesis de precedentes relevantes")
            st.markdown(res.sintesis)
        nombre = re.sub(r"\W+", "_", res.consulta[:40]).strip("_").lower()
        st.download_button("⬇️ Descargar informe completo (.md)", res.markdown(),
                           file_name=f"informe_{nombre}_{datetime.now():%Y%m%d_%H%M}.md", mime="text/markdown")
    with tab_p:
        for p in res.precedentes:
            c = p.candidato
            with st.expander(f"📜 Precedente {p.numero} · {c.doc_id}", expanded=p.numero == 1):
                a, b, cc, d = st.columns(4)
                a.markdown(f"**Materia**  \n{c.materia.upper()}")
                b.markdown(f"**Fecha**  \n{c.fecha}")
                cc.markdown(f"**Similitud**  \n{c.similitud:.3f}")
                d.markdown(f"**Fuente**  \n{p.fuente}{' · extracción genérica' if p.fallback else ''}")
                st.caption(f"Tipo indexado: {c.tipo}")
                st.markdown(p.informe)
                with st.popover("Ver hechos extraídos y verificaciones"):
                    st.markdown("**Hechos extraídos (paso de extracción):**")
                    st.markdown(p.extraccion)
                    if p.veredicto_relevancia:
                        st.markdown("**QC_1 (relevancia):**")
                        st.text(p.veredicto_relevancia)
                    if p.veredicto_reporte:
                        st.markdown("**QC_2 (anti-alucinación):**")
                        st.text(p.veredicto_reporte)
        if res.descartados:
            st.markdown("**Documentos descartados por los controles de calidad**")
            st.dataframe(res.descartados, width="stretch", hide_index=True)
    with tab_c:
        st.markdown("Top de categorías según el **score híbrido** (Coseno · Jaccard · KEM · Voto de recuperación · Concisión). "
                    "Las 3 primeras se usan como filtro estricto de la búsqueda.")
        st.dataframe(res.tabla_clasificacion.drop(columns=["tipo"]), width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["score_híbrido", "coseno", "jaccard", "kem", "voto_recuperación", "concisión", "penalidad"]})
    with tab_l:
        st.code("\n".join(st.session_state.get("logs", [])), language=None)

st.caption("Herramienta de apoyo basada en sentencias públicas del STJ Jalisco. No sustituye la asesoría de un "
           "profesional del derecho.")
