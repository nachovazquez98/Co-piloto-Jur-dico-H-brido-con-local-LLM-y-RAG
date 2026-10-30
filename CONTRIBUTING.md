# Guía de contribución

¡Gracias por tu interés en mejorar el Co-piloto Jurídico Híbrido!

## Preparar el entorno

```bash
git clone https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG.git
cd Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
ollama pull llama3.1:8b
pytest
streamlit run app/Copiloto.py
```

## Flujo de trabajo

1. Abre un *issue* que describa el error o la propuesta antes de cambios grandes.
2. Crea una rama desde `main`: `feature/<descripcion>` o `fix/<descripcion>`.
3. Haz commits pequeños con mensajes descriptivos.
4. Comprueba que `pytest` pasa y que la app arranca (`streamlit run app/Copiloto.py`).
5. Abre un *pull request* explicando qué cambia, por qué y cómo lo probaste.

## Estilo de código

- Python 3.10+, con *type hints* en las funciones públicas.
- Identificadores y comentarios siguen el estilo existente: nombres de funciones en inglés o español según el módulo,
  comentarios, logs y textos de la interfaz en **español**.
- Líneas de hasta 120 caracteres.
- La lógica va en `src/copiloto_juridico/`; `app/` solo contiene interfaz.
- El pipeline se comunica con la interfaz mediante eventos (`pipeline.Evento`): no llames a Streamlit desde el paquete.

## Datos

- **No subas PDFs** (`data/raw/` está ignorado) ni la base `data/chroma/` (se reconstruye sola).
- Si cambias el corpus o los parámetros de embeddings, regenera el índice con
  `python -m copiloto_juridico.cli indexar` e incluye `data/corpus/` y `data/indice/` en el commit.
- Mantén cada archivo por debajo de 50 MB (los fragmentos se dividen automáticamente).
- Respeta los términos del portal del STJ Jalisco; no se aceptan contribuciones que evadan sus controles de acceso
  (p. ej. reCAPTCHA).

## Reportar errores

Incluye: pasos para reproducirlo, consulta usada, modelo de LLM, sistema operativo / GPU y el contenido de la
pestaña **Logs** de la app.
