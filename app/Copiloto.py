"""Co-piloto Jurídico — interfaz web (Streamlit).

Ejecutar desde la raíz del repositorio:
    streamlit run app/Copiloto.py
"""

from __future__ import annotations

import html
import random
import re
import time
from datetime import datetime

import streamlit as st

from copiloto_juridico import cache_resultados
from copiloto_juridico.ayudas import AYUDA, AVISO_LEGAL
from copiloto_juridico.ejemplos import EJEMPLOS
from copiloto_juridico.fichas import extraer_articulos
from copiloto_juridico.llm import get_client
from copiloto_juridico.evaluacion.familias import etiqueta_familia
from copiloto_juridico.pipeline import PASOS, Copiloto, Opciones, Resultado, familias_disponibles, sugerir_familias

st.set_page_config(page_title="Co-piloto Jurídico", page_icon="⚖️", layout="wide")

from ui_common import get_clasificador, get_store, inject_css, sidebar_configuracion  # noqa: E402

inject_css()
settings, opciones = sidebar_configuracion()
store = get_store()
clasificador = get_clasificador()

st.title("⚖️ Co-piloto Jurídico")
st.markdown("Recuperación y análisis de precedentes de las Salas Penales del Supremo Tribunal de Justicia del Estado "
            "de Jalisco para la defensa en el recurso de apelación. Describa los hechos y la resolución que se "
            "impugna; el sistema localiza las resoluciones más afines y elabora una ficha de cada una. La página "
            "**Cómo funciona** explica el método.")

# --- Consulta ---
ESCRIBIR = "Escribir mi propia consulta"
por_etiqueta = {e.etiqueta: e for e in EJEMPLOS}


def _cargar_ejemplo() -> None:
    ej = por_etiqueta.get(st.session_state["ejemplo"])
    st.session_state["consulta"] = ej.texto if ej else ""


def _otro_ejemplo() -> None:
    st.session_state["ejemplo"] = random.choice([k for k in por_etiqueta if k != st.session_state.get("ejemplo")])
    _cargar_ejemplo()


if "consulta" not in st.session_state:
    st.session_state["consulta"] = ""
col_ej, col_otro = st.columns([5, 1], vertical_alignment="bottom")
col_ej.selectbox("Ejemplo (opcional)", [ESCRIBIR, *por_etiqueta], key="ejemplo", on_change=_cargar_ejemplo)
col_otro.button("Otro ejemplo", on_click=_otro_ejemplo, width="stretch",
                help="Carga al azar otro de los ejemplos de la lista")
consulta = st.text_area("Descripción del caso", key="consulta", height=140,
                        placeholder="Hechos, delito imputado, resolución que se apela y motivos de inconformidad...")


@st.cache_data(show_spinner=False)
def _familias(huella: int) -> dict[str, int]:
    f = familias_disponibles(store.corpus)
    return dict(sorted(f.items(), key=lambda kv: (kv[0] == "otros", kv[0])))  # "otros" al final


@st.cache_data(show_spinner=False, max_entries=200)
def _sugerencia(texto: str, huella: int) -> list[str]:
    return sugerir_familias(store, clasificador, texto)


familias = _familias(len(store.corpus))
sugerencia = [f for f in _sugerencia(consulta.strip(), len(store.corpus)) if f in familias] if consulta.strip() else []
if sugerencia:
    a, b = st.columns([4, 1], vertical_alignment="center")
    a.caption("Sugerencia inicial de la app según la descripción: "
              + ", ".join(f"**{etiqueta_familia(f)}** ({familias[f]})" for f in sugerencia)
              + ". Es una orientación: puede aceptarse, modificarse o dejar el delito sin seleccionar.")
    b.button("Usar la sugerencia", on_click=lambda: st.session_state.update(familias=sugerencia))
elegidas = st.multiselect("Delito (opcional): uno o varios", list(familias), key="familias",
                          format_func=lambda f: f"{etiqueta_familia(f)} ({familias[f]})",
                          placeholder="Sin selección: el sistema lo deduce de la descripción", help=AYUDA["delito"])
opciones["familias"] = tuple(elegidas)
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
            cache_resultados.guardar(ev.datos["resultado"], logs, settings, opciones)
        st.session_state["logs"] = logs
        ph_pasos.markdown(render_pasos(estado), unsafe_allow_html=True)


if buscar:
    st.session_state.pop("resultado", None)
    st.session_state.pop("guardado_el", None)
    guardado = cache_resultados.buscar(consulta.strip(), settings, opciones)
    if guardado:
        st.session_state.update(resultado=guardado["resultado"], logs=guardado["logs"], guardado_el=guardado["fecha"])
    else:
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
                       "candidatos. Revíselo con especial cuidado.")
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
    if st.session_state.get("guardado_el"):
        st.caption(f"Resultado guardado el {st.session_state['guardado_el'].replace('T', ' ')}")
    st.subheader("Síntesis")
    if res.aviso_cobertura:
        st.warning(res.aviso_cobertura)
    if res.nota_delito:
        st.info(res.nota_delito)
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
        st.markdown("**Clasificación de la consulta** " + (
            f"(la búsqueda se filtró por los delitos seleccionados: {', '.join(map(etiqueta_familia, res.familias_elegidas))})"
            if res.familias_elegidas else "(las 3 primeras categorías filtran la búsqueda)"))
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
