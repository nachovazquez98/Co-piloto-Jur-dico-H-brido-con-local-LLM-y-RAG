"""Página "Cómo funciona": el flujo, los límites de tamaño, el glosario y la última evaluación guardada."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from copiloto_juridico import config
from copiloto_juridico.ayudas import AVISO_LEGAL, FUNDAMENTOS, GLOSARIO
from copiloto_juridico.evaluacion.reporte import ultimo_resultado
from copiloto_juridico.pipeline import DESCRIPCION_PASOS, PASOS

st.set_page_config(page_title="Cómo funciona · Co-piloto Jurídico", page_icon="⚖️", layout="wide")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_common import inject_css  # noqa: E402

inject_css()
llm = config.LLMSettings()

st.title("Cómo funciona")
st.markdown("Esta página documenta el método del Co-piloto Jurídico: qué información utiliza, cómo la analiza y "
            "cómo deben interpretarse sus resultados.")

st.header("Objeto y alcance")
st.markdown("El sistema apoya a la defensa penal en la preparación o contestación del recurso de apelación ante las "
            "Salas Penales del Supremo Tribunal de Justicia del Estado de Jalisco. A partir de la descripción de un "
            "caso, localiza las resoluciones públicas del propio tribunal que resolvieron supuestos afines y elabora "
            "una ficha de cada una: delito, acto impugnado, apelante, agravios y su calificación, sentido del fallo "
            "y fundamento legal. El corpus comprende exclusivamente resoluciones en **materia penal**.\n\n"
            "El sistema **no** pronostica el resultado del asunto, no redacta agravios y no sustituye la lectura "
            "íntegra de la resolución ni el criterio del abogado.")

st.header("Guía de uso")
st.markdown("""
1. **Describa el caso** en la pantalla principal: los hechos, el delito imputado, la resolución que se impugna y
   los motivos de inconformidad. También puede partir de uno de los ejemplos.
2. **Delito (opcional).** El sistema sugiere los delitos que deduce de la descripción. La sugerencia es una
   orientación: puede aceptarse, modificarse, seleccionar varios o dejar el campo vacío para que el sistema lo
   determine. Si la selección difiere de lo que indican los hechos, el sistema lo advierte.
3. **Buscar precedentes.** El avance se muestra en cinco pasos; con fichas precalculadas, el análisis tarda
   alrededor de medio minuto.
4. **Síntesis.** Indica qué precedentes fueron favorables o desfavorables para la defensa y qué argumentos acogió o
   desestimó la Sala, con la cita de cada precedente ([P1], [P2]…).
5. **Tarjetas de precedentes.** Cada una contiene la ficha de la resolución. La leyenda **verificar** indica que un
   dato de la ficha no coincide con el texto (sentido del fallo distinto de los puntos resolutivos, o artículo
   citado que no aparece en la sentencia): consulte la resolución original antes de citarla.
6. **Informe.** Puede descargarse en Markdown. En *Detalles técnicos* se muestran la clasificación, los puntajes y
   el registro completo del análisis.
""")

st.header("Flujo del análisis")
for n, (clave, titulo) in enumerate(PASOS, 1):
    st.markdown(f"**{n}. {titulo}.** {DESCRIPCION_PASOS[clave]}")

st.header("Fundamentos técnicos")
for titulo, texto in FUNDAMENTOS.items():
    with st.expander(titulo):
        st.markdown(texto)

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
st.markdown("No se contó con abogados para anotar resultados, por lo que la referencia de la evaluación se construyó "
            "de forma automática e independiente del sistema: el delito, a partir de la categoría que asignó el "
            "propio STJ en su portal; el acto impugnado, el apelante y el sentido del fallo, por mayoría entre reglas "
            "aplicadas al texto y un modelo evaluador de otra familia (`gemma2:9b`), admitido solo tras acreditar "
            "al menos 0.80 de exactitud contra la categoría del STJ. Las métricas finales se miden una sola vez, con "
            "el código y los datos congelados. A continuación, la última corrida guardada de cada experimento "
            "(`evaluacion/resultados/`); lo no medido aparece como **por medir**.")
for comando, titulo in (("recuperacion", "Clasificación y recuperación"), ("qc", "Controles de calidad"),
                        ("fichas-referencia", "Calidad de la ficha"), ("esfuerzo", "Esfuerzo para encontrar precedentes"),
                        ("latencia", "Tiempo por consulta")):
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

st.header("Limitaciones")
st.markdown("""
- La referencia de la evaluación proviene del tribunal (su categoría) y del texto de la resolución, no del juicio de
  un experto sobre la utilidad de cada precedente.
- Algunos tipos penales cuentan con pocas resoluciones; cuando ningún precedente corresponde al delito detectado,
  el sistema lo advierte.
- Una sentencia extensa no cabe completa en la ventana del modelo: se envían el inicio, el estudio de los agravios
  y los puntos resolutivos.
- Las fichas las genera un modelo de lenguaje y pueden contener errores; por ello se verifican contra el texto y se
  señalan las discrepancias.
- Los documentos escaneados sin capa de texto no se aprovechan (no se aplica OCR).
""")

st.divider()
st.caption(AVISO_LEGAL)
