# Descarga de las sentencias (respaldo)

Cómo se obtuvieron los PDF del corpus desde el portal público del Supremo Tribunal de Justicia del Estado de Jalisco
(https://publicacionsentencias.stjjalisco.gob.mx/sentencias). El proyecto no necesita volver a descargarlos: el texto
depurado está en `data/corpus/` y los 1,899 PDF bajados a mano están como adjuntos de la versión
[`v11.2`](https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG/releases/tag/v11.2).

## Archivos

| Archivo | Contenido |
|---|---|
| `descargador_web.ipynb` | Copia sin modificar de las dos celdas de descarga del notebook original (commit `8d88a36`) |
| `delitos_penales.csv` | Los 480 delitos penales del catálogo del portal: familia, sentencias publicadas, en el corpus y cuántas faltan para el tope de 30 |
| `../evaluacion/descargas/` | Listas de descarga manual: `1_incompletas.csv`, `2_nuevas.csv`, `3_volver_a_bajar.csv` y su resumen `LISTA.md` |

## Las dos celdas del descargador

1. **API pública (hasta octubre de 2026).** Recorre la API `publica-sentencias-backend.stjjalisco.gob.mx` página
   por página y guarda cada PDF con el nombre `fecha-hora_materia_delito.pdf`. Desde octubre de 2026 la API exige
   reCAPTCHA y esta celda ya no descarga.
2. **Portal con verificación humana (Playwright).** Abre una ventana visible de Chromium en el portal; **una persona
   resuelve el reCAPTCHA** en esa ventana y el script continúa con esa sesión, con pausas entre descargas (1.5 s) y
   espera de 30 s si el servidor responde 429. Si el portal vuelve a pedir la verificación, el script se detiene
   hasta que la persona la resuelva otra vez. El script no resuelve ni evita el reCAPTCHA.

Requisitos de la celda 2: `pip install playwright` y `playwright install chromium`. Antes de ejecutarla, cambiar
`TARGET_FOLDER` por la carpeta de destino (en la copia es la ruta del equipo original).

## Procedimiento

1. Elegir qué bajar: `delitos_penales.csv` (columna «faltan para 30») o las listas de `evaluacion/descargas/`,
   ordenadas por familia con menos sentencias.
2. Descargar con la celda 2 o a mano desde el portal, buscando por delito. Cada PDF se nombra con la fecha y hora de
   publicación, la materia y el delito (`2025-03-20-115105_penal_violacion.pdf`).
3. Dejar los PDF en `data/raw/` y validarlos e incorporarlos al corpus:

   ```bash
   python -m copiloto_juridico.cli construir-corpus --pdfs data/raw/nuevas
   ```

   `construir-corpus` lee el PDF completo, recorta y limpia el texto, rechaza el que no corresponde a la misma
   resolución y suprime repetidas; después `python -m copiloto_juridico.cli indexar` actualiza el índice.

El catálogo del portal (`data/catalogo/catalogo_tipos_stj.csv`) es anterior a parte de las descargas: 2,400 de las
2,495 sentencias del corpus pertenecen a delitos que aparecen en él.
