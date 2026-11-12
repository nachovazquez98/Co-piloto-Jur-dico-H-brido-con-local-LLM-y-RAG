"""Componentes compartidos de la interfaz: recursos en caché, barra lateral del LLM y estilos."""

from __future__ import annotations

import logging

import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.clasificador import ClasificadorHibrido, construir_clasificador
from copiloto_juridico.llm import PROVIDER_DEFAULT_URLS, PROVIDERS, get_client
from copiloto_juridico.vectorstore import SentenciasStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CSS = """
<style>
:root { --cj-accent: #1f4e79; --cj-ok: #2e7d32; --cj-err: #c62828; --cj-muted: #8a8f98; }
.cj-hero h1 { margin-bottom: 0.1rem; }
.cj-hero p { color: var(--cj-muted); margin-top: 0; }
.cj-steps { border-left: 2px solid rgba(128,128,128,.25); margin: .5rem 0 1rem .6rem; padding-left: 1.1rem; }
.cj-step { position: relative; padding: .35rem 0 .55rem 0; }
.cj-step .dot { position: absolute; left: -1.62rem; top: .55rem; width: .95rem; height: .95rem; border-radius: 50%;
  background: rgba(128,128,128,.35); border: 2px solid transparent; box-sizing: border-box; }
.cj-step .title { font-weight: 600; }
.cj-step .detail { color: var(--cj-muted); font-size: .88rem; margin-top: .1rem; }
.cj-step .desc { color: var(--cj-muted); font-size: .8rem; margin-top: .05rem; line-height: 1.3; }
.cj-step .time { color: var(--cj-muted); font-size: .8rem; margin-left: .4rem; font-weight: 400; }
.cj-step.pending .title { color: var(--cj-muted); font-weight: 500; }
.cj-step.running .dot { background: transparent; border-color: var(--cj-accent); border-top-color: transparent;
  animation: cj-spin .8s linear infinite; }
.cj-step.running .title { color: var(--cj-accent); }
.cj-step.done .dot { background: var(--cj-ok); }
.cj-step.error .dot { background: var(--cj-err); }
.cj-step.error .title { color: var(--cj-err); }
@keyframes cj-spin { to { transform: rotate(360deg); } }
@media (prefers-color-scheme: dark) { :root { --cj-accent: #6fa8dc; } }
.cj-chip { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem;
  background: rgba(31,78,121,.12); color: var(--cj-accent); margin-right: .3rem; }
.cj-chip.cj-warn { background: rgba(198,40,40,.12); color: var(--cj-err); }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparando la base de sentencias (solo la primera vez tarda ~1 minuto)...")
def get_store() -> SentenciasStore:
    return SentenciasStore()


@st.cache_resource(show_spinner="Cargando el clasificador...")
def get_clasificador() -> ClasificadorHibrido:
    store = get_store()
    return construir_clasificador(store.corpus["tipo"].unique().tolist())


@st.cache_data(ttl=30, show_spinner=False)
def _listar_modelos(provider: str, base_url: str, api_key: str) -> list[str]:
    s = config.LLMSettings(provider=provider, base_url=base_url, api_key=api_key)
    return get_client(s).list_models()


def sidebar_configuracion() -> tuple[config.LLMSettings, dict]:
    """Barra lateral: solo el estado del modelo; todo lo demás en un desplegable "Configuración" cerrado."""
    defaults = config.LLMSettings()
    estado = st.sidebar.empty()
    with st.sidebar.expander("Configuración", expanded=False):
        opciones = {
            "num_precedentes": st.slider("Precedentes a analizar", 1, 5, config.RAG_NUM_RESULTS_DESIRED),
            "qc_relevancia": st.toggle("QC_1: verificar que el documento sirva como precedente", value=True),
            "qc_post_reporte": st.toggle("QC_2: verificar la ficha", value=True),
        }
        claves = list(PROVIDERS)
        provider = st.selectbox("Proveedor del modelo", claves,
                                index=claves.index(defaults.provider) if defaults.provider in claves else 0,
                                format_func=lambda k: PROVIDERS[k][0], key="llm_provider")
        default_url = defaults.base_url if provider == defaults.provider else PROVIDER_DEFAULT_URLS[provider]
        base_url = st.text_input("URL del servidor", value=default_url, key=f"llm_url_{provider}")
        api_key = ""
        if provider != "ollama":
            api_key = st.text_input("API key", value=defaults.api_key, type="password", key="llm_key")
        modelos = _listar_modelos(provider, base_url, api_key)
        default_model = defaults.model if provider == defaults.provider else ""
        if modelos:
            idx = modelos.index(default_model) if default_model in modelos else 0
            model = st.selectbox("Modelo", modelos, index=idx, key=f"llm_model_{provider}")
        else:
            model = st.text_input("Modelo", value=default_model or "gpt-4o-mini", key=f"llm_model_txt_{provider}")
        temperature = st.slider("Temperatura", 0.0, 1.0, float(defaults.temperature), 0.05)
        num_ctx = st.select_slider("Ventana de contexto (tokens)", [4096, 8192, 12288, 16384, 32768],
                                   value=defaults.num_ctx)
        max_tokens = st.slider("Máx. tokens de respuesta", 256, 4096, int(defaults.max_tokens), 128)
    settings = config.LLMSettings(provider=provider, model=model, base_url=base_url, api_key=api_key,
                                  temperature=temperature, num_ctx=num_ctx, max_tokens=max_tokens)
    ok, msg = get_client(settings).health() if model else (False, "Indica un modelo")
    estado.markdown(f"✓ Modelo listo: `{model}`" if ok else f"✗ {msg}")
    return settings, opciones
