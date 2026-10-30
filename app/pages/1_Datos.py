"""Página de datos: balance del corpus por materia y tipo, y herramientas para ampliarlo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.clasificador import etiqueta_tipo
from copiloto_juridico.datos.balance import balance_report, load_catalogo

st.set_page_config(page_title="Datos · Co-piloto Jurídico", page_icon="📊", layout="wide")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_common import get_store, inject_css  # noqa: E402

inject_css()
store = get_store()
corpus = store.corpus

st.title("📊 Corpus de sentencias")
st.markdown(
    f"El corpus incluido en el repositorio está **balanceado por tipo de sentencia** (materia + delito/acción): "
    f"cada tipo aporta como máximo **{config.MAX_SENTENCIAS_POR_TIPO} sentencias**, elegidas repartidas en el tiempo. "
    "Así la aplicación puede responder sobre cualquier tipo sin que los más frecuentes dominen los resultados.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sentencias", f"{len(corpus):,}")
c2.metric("Tipos distintos", corpus["tipo"].nunique())
c3.metric("Fragmentos indexados", f"{store.manifest['num_chunks']:,}")
c4.metric("Modelo de embeddings", store.manifest["embedding_model"].split("/")[-1])

col_g, col_t = st.columns([1, 1.3])
with col_g:
    st.subheader("Por materia")
    por_materia = corpus.groupby("materia").agg(sentencias=("id", "size"), tipos=("tipo", "nunique"))
    st.bar_chart(por_materia["sentencias"], horizontal=True, height=260)
    st.dataframe(por_materia, width="stretch")
with col_t:
    st.subheader("Por tipo de sentencia")
    tipos = store.tipos()
    tipos["tipo de sentencia"] = tipos["tipo"].map(etiqueta_tipo)
    materia_sel = st.multiselect("Filtrar materia", sorted(tipos["materia"].unique()))
    buscar = st.text_input("Buscar tipo", placeholder="p. ej. robo, divorcio, pagaré")
    vista = tipos
    if materia_sel:
        vista = vista[vista["materia"].isin(materia_sel)]
    if buscar:
        vista = vista[vista["tipo"].str.contains(buscar.lower().replace(" ", "-"), regex=False)]
    st.dataframe(vista[["tipo de sentencia", "sentencias"]].sort_values("sentencias", ascending=False),
                 width="stretch", hide_index=True, height=330,
                 column_config={"sentencias": st.column_config.ProgressColumn(
                     min_value=0, max_value=config.MAX_SENTENCIAS_POR_TIPO, format="%d")})

st.subheader("Balance frente al catálogo oficial del STJ")
rep = balance_report(corpus, load_catalogo())
completos, total = int(rep["completo"].sum()), len(rep)
st.progress(completos / total, text=f"{completos} de {total} tipos alcanzan su meta "
                                    f"(min({config.MAX_SENTENCIAS_POR_TIPO}, sentencias publicadas))")
st.caption("Los tipos incompletos solo pueden completarse descargando más sentencias del STJ. "
           f"Faltan {int(rep['faltan'].sum()):,} sentencias para el balance total.")
with st.expander("Ver tipos incompletos"):
    st.dataframe(rep[~rep["completo"]][["materia", "tipo", "en_corpus", "meta", "faltan"]],
                 width="stretch", hide_index=True)
    st.download_button("Descargar reporte de balance (.csv)", rep.to_csv(index=False), "balance_corpus.csv")

st.divider()
st.subheader("🛠️ Ampliar el corpus (opcional)")
st.markdown(
    "La aplicación ya funciona con el corpus incluido. Estas herramientas sirven para añadir sentencias y "
    "reconstruir el índice. Cada proceso muestra sus logs en vivo.")
st.warning("La API pública del STJ Jalisco exige reCAPTCHA desde 2026; la descarga automática se detendrá con un "
           "aviso si sigue activo. Alternativa: copia PDFs (nombrados `fecha_materia_delito.pdf`) a la carpeta de "
           f"PDFs (`{config.RAW_PDF_DIR}`) y procésalos.", icon="⚠️")


def correr(args: list[str]) -> None:
    cmd = [sys.executable, "-u", "-m", "copiloto_juridico.cli", *args]
    st.code(" ".join(cmd[2:]), language="bash")
    with st.container(height=320):
        ph = st.empty()
    lineas: list[str] = []
    proc = subprocess.Popen(cmd, cwd=config.ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for linea in proc.stdout:
        if "fontTools" in linea or "Warning" in linea:
            continue
        lineas.append(linea.rstrip())
        ph.code("\n".join(lineas[-300:]), language=None)
    proc.wait()
    if proc.returncode == 0:
        st.success("Proceso terminado.")
    else:
        st.error(f"El proceso terminó con código {proc.returncode}.")


a, b, c = st.columns(3)
with a:
    st.markdown("**1. Descarga balanceada**  \nDescarga solo los tipos que no han llegado a su meta y se detiene "
                "cuando todos están balanceados.")
    descargar = st.button("Descargar sentencias")
with b:
    st.markdown("**2. Procesar PDFs**  \nExtrae y limpia el texto (muestreo de páginas en PDFs largos) y lo une al "
                "corpus respetando el tope por tipo.")
    carpeta = st.text_input("Carpeta de PDFs", value=str(config.RAW_PDF_DIR))
    procesar = st.button("Procesar PDFs")
with c:
    st.markdown("**3. Reindexar**  \nRegenera los embeddings del corpus. Tarda unos minutos con GPU. "
                "Al terminar, reinicia la app.")
    reindexar = st.button("Reindexar corpus")

if descargar:
    correr(["descargar"])
if procesar:
    correr(["construir-corpus", "--pdfs", carpeta])
if reindexar:
    correr(["indexar"])
    st.cache_resource.clear()
