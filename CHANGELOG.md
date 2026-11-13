# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado [SemVer](https://semver.org/lang/es/).

## [11.0.0] - 2026-11-12

Respuesta a la evaluación del jurado: alcance acotado a la defensa penal en apelación, evaluación reproducible y
explicación de cada número. Ningún resultado se reporta si no está en una corrida guardada en `evaluacion/resultados/`.

### Añadido
- Módulo `evaluacion`: métricas (Precision/Recall/Hit@k, MRR, nDCG, F1 macro, kappa, IC de Wilson y bootstrap
  pareado) probadas con los ejemplos resueltos del informe; familias de delito (289 tipos penales -> 24 familias,
  CSV para revisión de un abogado); conjuntos de consultas, juicios y fichas con validación y detección de fuga;
  experimentos `recuperacion`, `ablacion`, `pesos`, `qc`, `qc-diagnostico`, `generacion`, `latencia`, `regresion`;
  borradores de consultas con el LLM (`revisado: false`, la partición de prueba se rechaza hasta revisarlos) y hojas
  de anotación CSV.
- Ficha penal orientada a la defensa: acto impugnado, apelante, agravios y su calificación, sentido del fallo con
  valores cerrados; generada en una sola llamada (JSON validado con reintento).
- Verificaciones deterministas de la ficha: sentido del fallo leído de los resolutivos, artículos citados contra el
  texto de la sentencia y QC_2 por familia del delito; las discrepancias se marcan "verificar". Si los
  resolutivos contradicen a la ficha, la ficha completa y el informe usan el sentido de los resolutivos.
- Síntesis v11: la lista de precedentes favorables o desfavorables para la defensa se calcula con reglas (quién
  apeló + sentido del fallo; también en cada tarjeta); el LLM resume los argumentos aceptados o rechazados y no
  pronostica el resultado (la v10 leyó al revés un precedente en una consulta real).
- Limpieza del texto: anexo de datos testados, etiquetas de testado, encabezados y pies de página aunque la
  extracción parta las palabras, números de expediente repetidos en cada hoja y palabras deletreadas; idempotente
  (`cli limpiar-corpus`, 16 % menos caracteres en el corpus guardado). Las sentencias penales se indexan completas.
  Comandos `cli pendientes` y `estado --salida`; lista de sentencias con páginas omitidas en
  `evaluacion/paginas_omitidas.csv`.
- Página "Cómo funciona" con el flujo, los límites de tamaño, glosario y la última evaluación guardada.

### Cambiado
- Por defecto: solo materia penal, clasificación por familia de delito, prompts v11; las opciones de v10 quedan por
  variable de entorno para comparar.
- Clasificador: Jaccard 0.40 y KEM 0 (eran redundantes); sin la lista fija de tokens penalizados; el embedding de
  cada familia incluye los nombres de sus tipos, sin el sufijo "y otro(s)" del portal ni nombres compuestos con
  otro delito; con una sola materia, "penal" ya no cuenta para Jaccard. Decisiones respaldadas por la regresión
  guardada.
- QC_1: lee el fragmento más parecido a la consulta y acepta mismo delito, misma familia o mismo problema
  procesal; si descarta todo, se muestran los mejores candidatos marcados "no verificado".
- Evaluación del conjunto A con *leave-one-out*: la sentencia fuente no vota por su propio tipo.
- Índice: sentencias penales completas (antes 12 fragmentos); PDF leídos completos (antes se muestreaban los de más
  de 50 páginas); recorte para el LLM que conserva inicio, análisis de agravios y resolutivos.
- Interfaz: pantalla principal con consulta, pasos, síntesis y tarjetas; detalles técnicos en un solo desplegable;
  configuración en la barra lateral, cerrada. Ayudas de una frase en cada número visible.
- LLM con semilla fija (42); embeddings en CPU si la GPU no tiene memoria.

### Eliminado
- Páginas "Datos" (sus herramientas están en la CLI) y "Evaluación" (integrada en "Cómo funciona").
- Ejemplos de otras materias en la app (se conservan como regresión en `evaluacion/regresion.py`).
- KEM sobre vocabulario y su archivo de términos (perdió en la regresión).

## [10.0.0] - 2026-10-30

### Añadido
- Interfaz web en Streamlit: consultas de ejemplo etiquetadas y editables, progreso animado por paso, logs en vivo,
  síntesis en streaming, resultados en pestañas y exportación a Markdown.
- Página **Datos** con el balance del corpus y herramientas de descarga, procesamiento y reindexación.
- Selector de LLM: Ollama o cualquier API compatible con OpenAI, configurable desde la interfaz o con `.env`.
- Búsqueda híbrida embeddings + BM25 con *Reciprocal Rank Fusion*.
- Voto de recuperación en la clasificación híbrida y filtro estricto por las 3 mejores categorías.
- Descargador balanceado (tope de 30 por tipo, validación de PDFs completos, detección de reCAPTCHA).
- Corpus balanceado (3,544 sentencias, 567 tipos) e índice de embeddings versionados en el repositorio.
- Paquete `copiloto_juridico`, CLI de datos, pruebas unitarias, `pyproject.toml`, `.env.example`.

### Cambiado
- Embeddings: `all-mpnet-base-v2` → `intfloat/multilingual-e5-base`, con *mean pooling* enmascarado,
  normalización L2 y fragmentación de cada sentencia.
- Las respuestas del LLM conservan mayúsculas y formato Markdown.
- Se fija `num_ctx` en Ollama y se recorta el contexto conservando el inicio y el final de cada sentencia.
- Las categorías del clasificador se derivan del corpus indexado.
- Notebooks movidos a `notebooks/` y documentos a `docs/`.

### Corregido
- QC_1 (relevancia) se había eliminado y aceptaba cualquier respuesta que contuviera "si"; QC_2 nunca se ejecutaba.
- `generate_final_synthesis` usaba marcadores que no existían en el prompt.

### Eliminado
- Cachés `rag_categories_cache7.json` / `rag_embeddings_cache7.npy` (no correspondían a la base en uso).

## [9.8] - 2025-11

- Versión en notebook (`notebooks/proyecto_8_llm_sentencia_judicial.ipynb`): pipeline RAG con clasificación híbrida,
  búsqueda adaptativa, extracción por materia y síntesis final.
