"""Co-piloto Jurídico — interfaz web (Streamlit).

Ejecutar desde la raíz del repositorio:
    streamlit run app/Copiloto.py
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime

import streamlit as st

from copiloto_juridico.ayudas import AYUDA, AVISO_LEGAL
from copiloto_juridico.ejemplos import EJEMPLOS
from copiloto_juridico.fichas import extraer_articulos
from copiloto_juridico.llm import get_client
from copiloto_juridico.pipeline import PASOS, Copiloto, Opciones, Resultado

st.set_page_config(page_title="Co-piloto Jurídico", page_icon="⚖️", layout="wide")

from ui_common import get_clasificador, get_store, inject_css, sidebar_configuracion  # noqa: E402

inject_css()
settings, opciones = sidebar_configuracion()
store = get_store()
clasificador = get_clasificador()

st.title("⚖️ Co-piloto Jurídico")
st.markdown("Describe tu caso y encuentra cómo han resuelto las Salas Penales del STJ Jalisco apelaciones parecidas.")

# --- Consulta ---
ESCRIBIR = "Escribir mi propia consulta"
por_etiqueta = {e.etiqueta: e for e in EJEMPLOS}


def _cargar_ejemplo() -> None:
    ej = por_etiqueta.get(st.session_state["ejemplo"])
    st.session_state["consulta"] = ej.texto if ej else ""


if "consulta" not in st.session_state:
    st.session_state["consulta"] = ""
st.selectbox("Ejemplo (opcional)", [ESCRIBIR, *por_etiqueta], key="ejemplo", on_change=_cargar_ejemplo)
consulta = st.text_area("Describe el caso", key="consulta", height=140,
                        placeholder="Hechos, delito, qué resolución se apela y por qué...")
buscar = st.button("Buscar precedentes", type="primary", disabled=not consulta.strip())


# --- Ejecución: solo los 5 pasos con su estado ---
def render_pasos(estado: dict) -> str:
    filas = []
    for clave, titulo in PASOS:
        e = estado[clave]
        tiempo = f"<span class='time'>{e['t']:.0f} s</span>" if e.get("t") is not None else ""
        detalle = f"<div class='detail'>{html.escape(e['detalle'])}</div>" if e.get("detalle") else ""
        filas.append(f"<div class='cj-step {e['estado']}'><span class='dot'></span>"
                     f"<div class='title'>{titulo}{tiempo}</div>{detalle}</div>")
    return "<div class='cj-steps'>" + "".join(filas) + "</div>"


def ejecutar(texto: str) -> None:
    copiloto = Copiloto(store, clasificador, get_client(settings))
    estado = {k: {"estado": "pending", "detalle": "", "t": None} for k, _ in PASOS}
    inicio: dict[str, float] = {}
    ph_pasos = st.empty()
    logs: list[str] = []
    for ev in copiloto.analizar(texto, Opciones(**opciones)):
        if ev.tipo == "paso_inicio":
            inicio[ev.paso] = time.time()
            estado[ev.paso]["estado"] = "running"
        elif ev.tipo == "paso_fin":
            estado[ev.paso].update(estado="done", detalle=ev.mensaje or estado[ev.paso]["detalle"],
                                   t=time.time() - inicio[ev.paso])
        elif ev.tipo == "progreso":
            estado[ev.paso]["detalle"] = f"{ev.mensaje} · fichas válidas {ev.datos['validos']}/{ev.datos['meta']}"
        elif ev.tipo == "log":
            logs.append(ev.mensaje)
            continue
        elif ev.tipo == "token":
            continue
        elif ev.tipo == "error":
            if ev.paso:
                estado[ev.paso].update(estado="error", detalle=ev.mensaje)
            st.error(ev.mensaje)
        elif ev.tipo == "resultado":
            st.session_state["resultado"] = ev.datos["resultado"]
        st.session_state["logs"] = logs
        ph_pasos.markdown(render_pasos(estado), unsafe_allow_html=True)


if buscar:
    st.session_state.pop("resultado", None)
    ejecutar(consulta.strip())
    if "resultado" in st.session_state:
        st.rerun()


# --- Resultado ---
def tarjeta(p) -> None:
    t = p.tarjeta()
    with st.container(border=True):
        st.markdown(f"#### Precedente {p.numero} · `{t['documento']}`")
        if t["no_verificado_qc1"]:
            st.warning("No verificado por QC_1: el control lo descartó, pero se muestra porque descartó todos los "
                       "candidatos. Revísalo con más cuidado.")
        a, b, c = st.columns(3)
        a.markdown(f"**Toca / expediente**  \n{t['identificador']}  \n**Fecha de publicación**  \n{t['fecha']}")
        b.markdown(f"**Delito**  \n{t['delito']}  \n**Acto impugnado**  \n{t['acto_impugnado']}")
        verificar_sentido = any(v.startswith("sentido") for v in t["verificar"])
        c.metric("Sentido del fallo", t["sentido_fallo"] + (" · verificar" if verificar_sentido else ""),
                 help=AYUDA["sentido_fallo"])
        if t["para_defensa"]:
            c.markdown(f"**Para la defensa:** {t['para_defensa']}", help=AYUDA["para_defensa"])
        if t["agravios"]:
            st.markdown("**Agravios y cómo los calificó la Sala**")
            st.markdown("\n".join(f"- {ag}" + (f" — **{cal}**" if cal else "") for ag, cal in t["agravios"]))
        if t["fundamento"]:
            faltan = set(t["articulos_no_encontrados"])
            st.markdown("**Fundamento legal**")
            st.markdown("\n".join(
                f"- {f}" + (" — _no encontrado en la sentencia, verificar_" if extraer_articulos(f) & faltan else "")
                for f in t["fundamento"]))
        for v in t["verificar"]:
            st.caption(f"Verificar: {v}")
        with st.expander("Ficha completa"):
            st.markdown(p.informe)


res: Resultado | None = st.session_state.get("resultado")
if res:
    st.subheader("Síntesis")
    if res.aviso_cobertura:
        st.warning(res.aviso_cobertura)
    st.markdown(res.sintesis)
    st.subheader("Precedentes")
    for p in res.precedentes:
        tarjeta(p)
    nombre = re.sub(r"\W+", "_", res.consulta[:40]).strip("_").lower()
    st.download_button("Descargar informe (.md)", res.markdown(),
                       file_name=f"informe_{nombre}_{datetime.now():%Y%m%d_%H%M}.md", mime="text/markdown")

    with st.expander("Detalles técnicos"):
        st.markdown(f"**Conceptos clave:** {res.entidades} · **Tiempo:** {res.segundos:.0f} s · "
                    f"**Llamadas al LLM:** {res.llamadas_llm} · **Modelo:** {res.modelo}")
        st.markdown("**Clasificación de la consulta** (las 3 primeras categorías filtran la búsqueda)")
        tabla = res.tabla_clasificacion[["categoría", "score_híbrido", "coseno", "jaccard", "voto_recuperación"]]
        st.dataframe(tabla, hide_index=True, width="stretch", column_config={
            "score_híbrido": st.column_config.NumberColumn("score", format="%.3f", help=AYUDA["score_hibrido"]),
            "coseno": st.column_config.NumberColumn(format="%.3f", help=AYUDA["coseno"]),
            "jaccard": st.column_config.NumberColumn(format="%.3f", help=AYUDA["jaccard"]),
            "voto_recuperación": st.column_config.NumberColumn("voto", format="%.3f", help=AYUDA["voto"])})
        st.markdown("**Puntajes de cada precedente**")
        st.dataframe([{"documento": p.candidato.doc_id, "similitud": p.candidato.similitud,
                       "RRF": p.candidato.relevancia, "fuente": p.fuente} for p in res.precedentes],
                     hide_index=True, width="stretch", column_config={
                         "similitud": st.column_config.NumberColumn(format="%.3f", help=AYUDA["similitud"]),
                         "RRF": st.column_config.NumberColumn(format="%.4f", help=AYUDA["rrf"]),
                         "fuente": st.column_config.Column(help=AYUDA["fuente"])})
        if res.descartados:
            st.markdown("**Candidatos descartados**", help=AYUDA["descartados"])
            st.dataframe(res.descartados, hide_index=True, width="stretch")
        if res.veredictos_ambiguos:
            st.caption(f"Veredictos de control no interpretables: {res.veredictos_ambiguos}. {AYUDA['ambiguos']}")
        st.markdown("**Registro**")
        st.code("\n".join(st.session_state.get("logs", [])), language=None)

st.caption(AVISO_LEGAL)
