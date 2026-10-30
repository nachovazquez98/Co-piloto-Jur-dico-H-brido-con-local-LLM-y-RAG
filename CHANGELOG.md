# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado [SemVer](https://semver.org/lang/es/).

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
