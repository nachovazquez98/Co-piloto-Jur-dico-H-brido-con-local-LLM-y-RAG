# Datos

| Carpeta | Contenido | ¿En git? |
|---|---|---|
| `corpus/sentencias-*.parquet` | Texto limpio y metadatos de cada sentencia (`id`, `fecha`, `materia`, `delito_o_accion`, `tipo`, `texto`, `origen`). Zstd. | Sí |
| `indice/` | `manifest.json`, `fragmentos.parquet` (fragmento → sentencia) y `embeddings-*.npy` (float16, 768 dim, `intfloat/multilingual-e5-base`). | Sí |
| `catalogo/catalogo_tipos_stj.csv` | Catálogo oficial de tipos (`Materia_DELITO[_ACCIÓN]`, nº de sentencias publicadas). Define la meta de balance de cada tipo. | Sí |
| `chroma/` | Base ChromaDB local. Se reconstruye sola al arrancar si cambia el índice. | No |
| `raw/pdfs/` | PDFs descargados o copiados a mano para ampliar el corpus. | No |

## Origen

Sentencias públicas del [Supremo Tribunal de Justicia del Estado de Jalisco](https://publicasentencias.stjjalisco.gob.mx),
descargadas de su API pública (fase 0 del notebook) y procesadas en la fase 1 (extracción con muestreo de páginas en
PDFs largos y limpieza por regex). El texto se recuperó de la base `chroma_db_juridica_8.2` (5,805 sentencias) y de 44
PDFs locales (ya incluidos en esa base).

## Criterio de balance

- **Tipo** = materia + delito/acción, tal como lo publica el STJ (p. ej. `penal/robo-calificado`).
- **Meta por tipo** = `min(30, sentencias publicadas de ese tipo)`.
- Si un tipo tiene más de 30, se eligen 30 **repartidas uniformemente en el tiempo**.
- La descarga balanceada termina cuando todos los tipos del catálogo alcanzan su meta.

Estado actual: 3,544 sentencias en 567 tipos. 273 tipos del catálogo alcanzan su meta; los demás están limitados por
las sentencias que se habían descargado (`python -m copiloto_juridico.cli estado`).
