# Evaluación

Todos los comandos: `python -m copiloto_juridico.evaluacion <comando>` desde la raíz del repositorio. Cada corrida
queda en `evaluacion/resultados/` (JSON y Markdown) con fecha, commit, configuración, semillas, digest del modelo y
huella del corpus. Las decisiones de diseño citan en `config.py` la corrida que las respalda.

## Referencia automática, sin anotadores humanos (v11.2)

No hay abogados disponibles para anotar, así que la referencia se construye sin intervención humana:

| Qué se necesita | De dónde sale |
|---|---|
| Delito de cada sentencia | La **categoría que asignó el STJ** en su portal (externa al sistema) → familia (`familias_penal.csv`) |
| Acto impugnado, apelante y sentido del fallo | Mayoría absoluta entre las reglas (`metadatos.py`) y el juez LLM; sin mayoría, "no determinado" y no cuenta |
| Relevancia de un par (consulta, sentencia), 0/1/2 | Mismo delito (familia de la categoría del STJ) y mismo problema procesal (mayoría entre reglas y juez): 2 = ambos, 1 = uno, 0 = ninguno |
| Consultas válidas | Comprobaciones de la guía (no copia la sentencia, no nombra el tipo del portal, sin datos identificables, no es una negativa del modelo) y aceptación del juez |
| Agrupación tipo → familia | `familias --coherencia`: centroides de embeddings por tipo y familia |
| Esfuerzo del usuario | `esfuerzo`: sentencias que hay que abrir para encontrar 3 útiles (portal, búsqueda por palabras, app) |

**Jueces.** Modelos de familias distintas a la del modelo de la app, uno a la vez en la GPU de 8 GB. Un juez solo
cuenta si acierta la familia de la categoría del STJ en al menos `JUEZ_EXACTITUD_MINIMA` = 0.80 de las sentencias,
criterio fijado antes de correrlos:

| Juez | Resultado | Evidencia |
|---|---|---|
| `qwen2.5:7b` | Descartado en la prueba piloto (60 sentencias): no llegó a 0.80 | comentario en `config.py` |
| `qwen2.5:14b` | Descartado por recursos: no cabe en 8 GB y la laptop no lo sostiene en tandas largas | comentario en `config.py` |
| `mistral:7b` | **No usable:** 0.795 en las 1,412 sentencias | `resultados/20261117-065346_a60ca4f_referencias` |
| `gemma2:9b` | **Usable:** 0.845, IC 95 % [0.807, 0.877], en la muestra de 401 | `resultados/20261117-065346_a60ca4f_referencias` |

**Muestra.** Juzgar todo con un juez tomaba ~7 h. Los jueces y la referencia usan una muestra fija por hash del id
con la semilla de evaluación (`config.MUESTRA_SENTENCIAS` = 0.30, `MUESTRA_CONSULTAS` = 0.50): al crecer el corpus,
los elementos que ya estaban no cambian. Una consulta entra con todos sus pares, para que las métricas por consulta
tengan todos sus juicios. `jueces --completa` y `referencias --completa` usan todo.

**Límite que hay que declarar:** es una evaluación contra lo que dicen el tribunal y el texto de la sentencia, leído
por reglas y por un LLM; no contra el juicio de un experto sobre la utilidad de un precedente.

## Orden de los comandos

```bash
copiloto-datos construir-corpus --pdfs CARPETA ...  # rechaza PDFs con el texto de otra sentencia y quita repetidas
copiloto-datos indexar                              # embeddings + metadatos procesales por reglas
fichas-batch                     # fichas de las sentencias nuevas, cambiadas o inválidas (llama3.1:8b) + diagnóstico
borradores --ampliar N           # (opcional) consultas nuevas hasta N por familia
conceptos                        # conceptos clave de las consultas (llama3.1:8b)
jueces --pares                   # pares (consulta, sentencia) a juzgar; sin LLM
jueces --modelo gemma2:9b        # el juez, sobre la muestra (reanudable)
referencias                      # referencias, juicios y consultas validadas; sin LLM
congelar
recuperacion --particion prueba
qc --particion prueba
fichas-referencia --particion prueba
esfuerzo --particion prueba
familias --coherencia
latencia --ejemplos --max 10
regresion                        # diagnóstico con los ejemplos de la app
errores --corrida ID [ID ...]    # un caso por fallo, con su tipo
```

Desarrollo antes de prueba: `recuperacion`, `ablacion`, `pesos` y `qc` con `--particion desarrollo` sirven para
elegir; la partición de prueba se mide una sola vez.

`errores` clasifica cada fallo en el primer paso del pipeline que falla: `cobertura` (no hay sentencias de ese tipo),
`clasificacion` (la familia correcta quedó fuera del top 3), `orden` (la familia estaba, pero el precedente relevante
quedó fuera del top k), `qc1_descarte` (QC_1 eliminó un candidato relevante), `ficha` (sentido o familia distintos
de la referencia) u `otro`.

## Corrida final: congelar y medir una sola vez

`congelar` guarda en `evaluacion/congelado.json` la huella del código, corpus, índice, fichas, configuración
efectiva, texto de cada prompt y digest del modelo. Cada corrida de prueba se niega a empezar si algo no coincide
(lo dice en una línea) y solo se permite **una vez** por comando y congelado (`prueba_corridas.jsonl` guarda la
marca). Repetirla exige `--forzar "motivo"`, y el motivo queda en el reporte. Las corridas de desarrollo solo avisan
si algo cambió. Solo cuentan las consultas validadas por `referencias`.

## Archivos

| Archivo | Qué es |
|---|---|
| `consultas_penal.jsonl` | Consultas redactadas por el LLM a partir de sentencias (`S…`: temas sin sentencias en el corpus) |
| `familias_penal.csv` | Tipo del portal → familia de delito |
| `regresion_conceptos.json`, `cache/conceptos.json` | Conceptos precalculados (la evaluación no llama al LLM para esto) |
| `jueces/` | Respuestas de cada juez (JSONL con la clave del prompt), pares, referencias, juicios y consultas validadas |
| `resultados/` | Todas las corridas |
| `errores/` | Análisis de errores y casos para la presentación |
| `descargas/` | Listas de sentencias por descargar a mano desde el portal (incompletas y nuevas) |
| `paginas_omitidas.csv` | Sentencias del corpus que todavía tienen páginas omitidas |
