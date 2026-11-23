# Descarga de las sentencias

Procedimiento para obtener los PDF de las sentencias desde el portal público del Supremo Tribunal de Justicia del
Estado de Jalisco (https://publicacionsentencias.stjjalisco.gob.mx/sentencias), de **cualquier materia**. El proyecto
no necesita volver a descargarlas: el texto depurado está en `data/corpus/` y los 1,899 PDF bajados a mano están como
adjuntos de la versión [`v11.2`](https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG/releases/tag/v11.2).

## Archivos

| Archivo | Contenido |
|---|---|
| `descargar.py` | Descargador de línea de comandos: todas las materias, filtros, tope por tipo, validación y reanudación |
| `descargar.ipynb` | Los mismos pasos desde un notebook |
| `delitos_penales.csv` | Los 480 delitos penales del catálogo: familia, sentencias publicadas, en el corpus y cuántas faltan para el tope de 30 |
| `../data/catalogo/catalogo_tipos_stj.csv` | Catálogo del portal de todas las materias: tipo (materia + delito o acción) y sentencias publicadas |
| `../evaluacion/descargas/` | Listas de descarga manual del corpus penal y su resumen `LISTA.md` |

Catálogo del portal por materia (`data/catalogo/catalogo_tipos_stj.csv`):

| Materia | Tipos | Sentencias publicadas |
|---|---|---|
| Penal | 487 | 20,808 |
| Civil | 207 | 12,404 |
| Mercantil | 50 | 4,129 |
| Familiar | 110 | 3,354 |

## Modos

- **`api`**: consulta directa a la API pública. Funcionó hasta octubre de 2026; desde entonces la API exige reCAPTCHA
  y el script se detiene con un aviso.
- **`navegador`**: abre una ventana visible de Chromium en el portal. **Una persona resuelve el reCAPTCHA** y el
  script continúa con esa sesión; si el portal vuelve a pedir la verificación, espera a que la persona la resuelva
  otra vez. Pausa de 1.5 s entre descargas y espera de 30 s si el servidor responde 429. El script nunca resuelve ni
  evita el reCAPTCHA.

Requisitos: `pip install requests playwright` y `playwright install chromium` (solo el modo navegador usa Playwright).

## Uso

```bash
# Corpus penal del proyecto: hasta 30 sentencias por tipo, contando las que ya tiene el corpus
python descarga_datos/descargar.py --destino data/raw/nuevas --materias penal --max-por-tipo 30 \
       --descontar-corpus --modo navegador

# Todas las materias, o algunas, o un delito o acción concretos
python descarga_datos/descargar.py --destino ~/sentencias --materias todas --modo navegador
python descarga_datos/descargar.py --destino ~/sentencias --materias civil,familiar --delito divorcio --modo navegador

# Solo la lista de asuntos publicados (catalogo_asuntos.csv), sin descargar PDF
python descarga_datos/descargar.py --destino ~/sentencias --solo-catalogo --modo navegador
```

Cada PDF se nombra con la fecha y hora de publicación, la materia y el delito o acción
(`2025-03-20-115105_penal_violacion.pdf`), se valida (cabecera `%PDF` y marcador `%%EOF`) y se escribe de forma
atómica. Una descarga interrumpida se reanuda: lo que ya está completo no se vuelve a pedir y el tope por tipo cuenta
los PDF de la carpeta. `registro_descargas.csv`, en la carpeta de destino, registra cada asunto con su estado
(descargado, ya estaba, tope alcanzado, PDF incompleto, error).

## Incorporar las sentencias al corpus

```bash
python -m copiloto_juridico.cli construir-corpus --pdfs data/raw/nuevas
python -m copiloto_juridico.cli indexar
```

`construir-corpus` lee el PDF completo, recorta y limpia el texto, rechaza el que no corresponde a la misma resolución
y suprime repetidas; `indexar` actualiza el índice. El método vigente del proyecto (familias de delito, ficha,
evaluación) es específico de la materia penal: las sentencias de otras materias pueden descargarse con este script,
pero la aplicación solo busca en las penales (`COPILOTO_MATERIAS`).

Las celdas originales del notebook del proyecto (fase 0) se conservan en el historial de git (commit `8d88a36`);
`descargar.py` las reúne con la ruta de destino como parámetro, el nombre de archivo que lee `construir-corpus`, la
validación de cada PDF y el registro.
