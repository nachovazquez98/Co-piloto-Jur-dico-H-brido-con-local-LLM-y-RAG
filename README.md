# Co-piloto Jurídico Híbrido (v9.8)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=yellow)
![spaCy](https://img.shields.io/badge/spaCy-v3.0%2B-brightgreen?logo=spacy)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-blueviolet)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)

Este repositorio contiene el código de un prototipo avanzado para un **Co-piloto Jurídico Híbrido**, un sistema de Generación Aumentada por Recuperación (RAG) diseñado para analizar precedentes legales (sentencias judiciales) de forma 100% local y privada.

El sistema es "híbrido" en dos sentidos:
1.  **Ejecución Híbrida:** Utiliza un modelo de embeddings para la *recuperación* de documentos y un LLM local (`llama3.1:8b` servido por `Ollama`) para la *generación* de análisis.
2.  **Clasificación Híbrida:** Implementa un novedoso clasificador numérico (`classify_dynamically`) que combina la búsqueda semántica (Coseno) con métricas léxicas (Jaccard, KEM) para categorizar la consulta del usuario, eliminando la necesidad de una llamada costosa al LLM en la etapa de filtrado.

## 🚀 Características Principales

* **100% Local y Privado:** Todo el pipeline, desde la indexación hasta la generación de informes, se ejecuta localmente. No se envían datos a APIs externas, garantizando la confidencialidad.
* **Ingesta Paralelizada:** La Fase 1 (`fase1_procesar_pdfs`) utiliza `concurrent.futures.ProcessPoolExecutor` para procesar múltiples PDFs en paralelo, acelerando significativamente la creación del corpus.
* **Clasificación Híbrida-Numérica:** Un clasificador rápido y ligero (`classify_dynamically`) que utiliza `spaCy` y `scikit-learn` para categorizar la consulta del usuario en una de las 1053 categorías legales sin usar un LLM.
* **Búsqueda Adaptativa:** El sistema primero intenta una "Búsqueda Estricta" (filtrada por la categoría predicha) y, si no encuentra suficientes resultados, automáticamente realiza una "Búsqueda Amplia" semántica (fallback).
* **Pipeline de Doble Control de Calidad (QC):**
    1.  **QC_1 (Relevancia de Contexto):** Un prompt de verificación (`PROMPT_VERIFICACION_RELEVANCIA`) se asegura de que el *documento* recuperado sea contextualmente relevante (ej. distingue "lesiones en riña" de "lesiones por violencia familiar").
    2.  **QC_2 (Anti-Alucinación):** Un segundo prompt (`PROMPT_VERIFICACION_POST_REPORTE`) verifica que el *informe generado* por el LLM sea coherente con la consulta original (ej. descarta un informe sobre "fraude" si la consulta era sobre "lesiones").
* **Extracción Robusta con Fallback:** El sistema primero intenta una extracción estructurada (`PROMPT_EXTRACCION_PENAL`). Si falla (ej. debido a texto anonimizado `[ELIMINADO]`), reintenta automáticamente con un prompt genérico (`PROMPT_EXTRACCION_GENERICA_FALLBACK`) para maximizar la tasa de éxito.
* **Síntesis Comparativa Final:** En lugar de solo listar documentos, el sistema genera un resumen ejecutivo (`PROMPT_SINTESIS_FINAL`) que compara los 3 precedentes válidos encontrados, ofreciendo un análisis de valor agregado.

## ⚙️ Stack Tecnológico

* **Servidor LLM:** `Ollama`
* **Modelo LLM:** `llama3.1:8b` (o cualquier modelo compatible con Ollama)
* **Base de Datos Vectorial:** `ChromaDB` (para almacenamiento persistente)
* **Modelo de Embeddings:** `sentence-transformers/all-mpnet-base-v2`
* **Procesamiento de PDF:** `PyPDF2`
* **Procesamiento de Texto y Métricas:** `spaCy` (lematización), `scikit-learn` (cosine_similarity)
* **Manejo de Datos:** `pandas`, `pyarrow` (para almacenamiento en Parquet)
* **Paralelización:** `concurrent.futures.ProcessPoolExecutor`

## 🏛️ Arquitectura y Flujo de Trabajo

El sistema opera en tres fases principales:

### Fase 1: Ingesta y Procesamiento (Paralelizado)

1.  **Escanear:** El script escanea `PDF_DATABASE_PATH` en busca de archivos `.pdf`.
2.  **Procesar en Paralelo:** `fase1_procesar_pdfs` usa `ProcessPoolExecutor` para distribuir la carga de trabajo.
3.  **Extraer y Limpiar:** Cada proceso hijo (`process_single_pdf`) abre un PDF, aplica un muestreo estratégico para documentos largos, y usa Regex (`RE_NOISE_F1`, `RE_STRUCTURAL_KEYWORDS_F1`) para extraer y limpiar el cuerpo legal del texto.
4.  **Parsear Metadatos:** Los metadatos (`materia_principal`, `delito_o_accion`) se extraen de la nomenclatura del nombre del archivo.
5.  **Guardar en Lotes:** Los datos limpios (ID, texto, metadatos) se guardan en archivos `.parquet` en el directorio `BATCH_TEMP_DIR`.

### Fase 2: Indexación en la Base de Datos Vectorial

1.  **Cargar Modelo:** Se inicializa el modelo de embeddings (`all-mpnet-base-v2`) y se mueve a la GPU (CUDA) si está disponible.
2.  **Conectar a DB:** Se inicializa `chromadb.PersistentClient` y se crea (o limpia) la colección `sentencias_judiciales`.
3.  **Procesar Lotes:** El script itera sobre los archivos `.parquet` de la Fase 1.
4.  **Generar Embeddings:** Los textos de cada lote se dividen en sub-lotes (`INDEXING_BATCH_SIZE = 50`) para generar los embeddings vectoriales sin sobrecargar la VRAM de la GPU.
5.  **Indexar:** Los embeddings, documentos (texto) y metadatos se cargan en `ChromaDB`.

### Fase 3: Inferencia RAG (El Co-piloto)

Este es el flujo de ejecución principal para cada consulta del usuario:

1.  **Cargar Caché:** Carga las 1053 categorías únicas y sus embeddings pre-calculados (`rag_categories_cache7.json`, `rag_embeddings_cache7.npy`).
2.  **Clasificación Híbrida:** `classify_dynamically` identifica la mejor categoría-filtro (ej. `PENAL robo-calificado...`) usando una fórmula ponderada de Coseno, Jaccard, KEM y Concisión.
3.  **Búsqueda Adaptativa:** `generate_multianalysis_report_from_rag` intenta una "Búsqueda Estricta" en ChromaDB usando la categoría como filtro `where`. Si falla, "relaja" la consulta y realiza una "Búsqueda Amplia" semántica.
4.  **Bucle de Generación (Doble QC + Fallback):**
    * El sistema itera sobre los 50 mejores candidatos (`RAG_NUM_RESULTS_TO_FETCH = 50`) hasta encontrar 3 válidos (`RAG_NUM_RESULTS_DESIRED = 3`).
    * **Pasa por QC_1:** `PROMPT_VERIFICACION_RELEVANCIA` comprueba si el *contexto* del documento coincide (ej. "riña de bar" vs "violencia familiar").
    * **Pasa por Extracción:** Intenta `PROMPT_EXTRACCION_PENAL`.
    * **Pasa por Fallback:** Si la extracción falla (ej. por texto anonimizado `[ELIMINADO]`), reintenta con `PROMPT_EXTRACCION_GENERICA_FALLBACK`.
    * **Pasa por QC_2:** `PROMPT_VERIFICACION_POST_REPORTE` comprueba que el *informe generado* no sea una alucinación (ej. un informe de "fraude" en un documento de "lesiones").
5.  **Síntesis Final:** Los 3 informes válidos se envían a `generate_final_synthesis`, que usa `PROMPT_SINTESIS_FINAL` para crear un resumen comparativo.
6.  **Entrega:** Se presenta al usuario la Síntesis (respuesta principal) y el Apéndice (los 3 informes detallados).

## 💡 Hallazgos Clave y Robustez del Sistema

Durante las pruebas, se identificaron varios puntos de fallo que esta arquitectura (v9.8) está diseñada para manejar:

* **Problema: Datos Mal Etiquetados.**
    * **Hallazgo:** Un documento sobre "acta de nacimiento" estaba incorrectamente etiquetado como "robo calificado" en el nombre del archivo.
    * **Solución:** El filtro **QC_1 (`PROMPT_VERIFICACION_RELEVANCIA`)** detectó esta discrepancia contextual y descartó el documento, evitando que contaminara los resultados.

* **Problema: Clasificación Incorrecta.**
    * **Hallazgo:** La consulta sobre "riña en un bar" fue clasificada erróneamente. Esto llevó a la Búsqueda Amplia, que recuperó documentos semánticamente similares pero contextualmente incorrectos (ej. "violencia familiar").
    * **Solución:** El filtro **QC_1** (con su ejemplo explícito "riña de bar vs. violencia familiar") detectó y descartó exitosamente estos falsos positivos.

* **Problema: Alucinación del LLM.**
    * **Hallazgo:** En un caso, el sistema recuperó un documento sobre "lesiones" (aprobado por QC_1), pero el LLM alucinó y generó un informe sobre "fraude".
    * **Solución:** El filtro **QC_2 (`PROMPT_VERIFICACION_POST_REPORTE`)** comparó el informe generado ("fraude") con la consulta original ("lesiones") y descartó el informe, previniendo una alucinación grave.

* **Problema: Texto Anonimizado.**
    * **Hallazgo:** El documento más relevante para la consulta de "riña" (`lesiones-en-rina.pdf`) estaba lleno de texto `[ELIMINADO]`, lo que provocó que el `PROMPT_EXTRACCION_PENAL` fallara.
    * **Solución:** La **lógica de fallback** se activa, reintentando con `PROMPT_EXTRACCION_GENERICA_FALLBACK`. Los prompts de extracción actualizados ahora contienen instrucciones explícitas para ignorar las marcas de anonimización y resumir la información visible.

## 🔧 Propuestas de Mejora (Trabajo Futuro)

1.  **Ingesta de Metadatos con IA:** El eslabón más débil sigue siendo la dependencia de la nomenclatura de archivos. La Fase 1 debería mejorarse para usar un LLM (`PROMPT_CLASIFICAR_MATERIA`) que lea el contenido de cada PDF y *genere* los metadatos de forma fiable.
2.  **Ajuste de Pesos del Clasificador:** Los pesos del re-ranking híbrido (ej. `JACCARD_WEIGHT`) son heurísticos. Se podrían ajustar o entrenar en un conjunto de datos de prueba para mejorar la precisión de la clasificación (Paso 2).
3.  **Integración de OCR:** Añadir `Tesseract` o `PyMuPDF` para manejar documentos que sean imágenes escaneadas, aumentando la cantidad de datos procesables.
