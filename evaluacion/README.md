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
| `gemma2:9b` | **Usable:** 0.845, IC 95 % [0.807, 0.877], en la muestra de 401 del corpus anterior | `resultados/20261117-065346_a60ca4f_referencias` |
| `gemma2:9b`, corpus final (25 familias) | **No usable:** 0.742, IC 95 % [0.709, 0.773], en las 725 sentencias de la muestra | `resultados/20261121-031724_723fe6b_referencias`; con 24 familias, 0.739 (`20261121-003449_5311dc1`), errores en `errores/juez_familia.md` |

**Consecuencia.** Sin juez usable, el acto impugnado, el apelante y el sentido de la referencia salen de las reglas, que
la app también usa para ordenar candidatos: esas métricas no son independientes y se reportan como tales. Las métricas
principales usan solo la familia de la categoría del STJ, externa al sistema.

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
latencia --ejemplos --max 10      # mismos 16 ejemplos de regresión
regresion                        # diagnóstico con los 16 ejemplos penales de regresión (ejemplos.EJEMPLOS_PENALES)
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

## Guía de interpretación de las métricas

Valores de la corrida final (partición de prueba, congelado `a15b83c852281010`, 46 consultas validadas).

| Métrica | Pregunta que responde | Cálculo | Resultado y lectura |
|---|---|---|---|
| **Hit@3 de familia** (`recuperacion`) | ¿El filtro de búsqueda contiene el delito correcto? | Fracción de consultas cuya familia del STJ está entre las 3 que elige el clasificador; la sentencia de origen no vota (*leave-one-out*); IC 95 % de Wilson | 0.717 [0.575, 0.827]: en 7 de cada 10 consultas el sistema busca en la familia correcta; la línea base k-NN, 0.435 |
| **Hit@k** | ¿Aparece la sentencia de la que se redactó la consulta? | Fracción de consultas con esa sentencia entre los k primeros | Hit@5 de la app 0.174; con re-ranker 0.370. Mide un solo documento: hay otros precedentes útiles |
| **MRR** | ¿Qué tan arriba aparece esa sentencia? | Promedio de 1/posición (1 = primera, 0.5 = segunda; lista de 30) | App 0.152; BM25 solo 0.233 |
| **nDCG@5** | ¿Los 5 primeros están bien ordenados por utilidad? | Ganancia con descuento por posición, normalizada por el orden ideal; relevancia 2 = mismo delito y mismo problema procesal, 1 = uno de los dos | App 0.285; BM25 0.243; embeddings 0.304; 1 = orden perfecto |
| **P@5** | ¿Cuántos de los 5 primeros son pertinentes? | Fracción del top 5 con relevancia ≥ 1 | App 0.265: 1.3 de los 5 primeros |
| **Diferencia pareada** | ¿Una variante supera a otra? | Media por consulta de (A − B), IC 95 % por *bootstrap* pareado (1,000 remuestreos); efecto mínimo detectable con alfa 5 % y potencia 80 % | Significativa si el intervalo no incluye 0: el híbrido supera a k-NN en Hit@3 (+0.283); el re-ranker mejora Hit@5 (+0.196) pero no nDCG@5 |
| **Exactitud de la ficha** (`fichas-referencia`) | ¿La ficha lee bien delito, acto, apelante y sentido? | Coincidencia con la referencia por campo; IC de Wilson | Familia 0.917, acto 0.883, apelante 0.830, sentido 0.722 |
| **Precisión del fundamento legal** | ¿Los artículos que cita la ficha son los que invocó la Sala? | Por ficha, citados que figuran en el fundamento ÷ citados; promedio con IC *bootstrap* | 0.644 [0.600, 0.690]: casi 2 de cada 3 artículos citados son fundamento de la Sala |
| **Exhaustividad del fundamento** | ¿Cuánto del fundamento recoge la ficha? | Citados en el fundamento ÷ artículos del fundamento | 0.133: la ficha resume los principales; la sentencia invoca muchos |
| **Artículos no encontrados** | ¿La ficha cita artículos ausentes de la sentencia? | Artículos de la ficha que no aparecen en el texto | 7 de 350 fichas; la app los marca con «verificar» |
| **Precisión / recall / kappa de QC_1** (`qc`) | ¿El control de relevancia descarta precedentes útiles? | Veredictos contra la referencia en 776 pares; kappa descuenta el acuerdo por azar | Recall 0.721: conserva 7 de cada 10 relevantes; precisión 0.322 y kappa 0.051: deja pasar muchos no relevantes: está diseñado para no descartar; el orden por facetas y QC_2 actúan después |
| **Esfuerzo** (`esfuerzo`) | ¿Cuántas sentencias hay que abrir para hallar 3 útiles? | Sentencias abiertas hasta reunir 3 útiles (misma familia y mismo acto impugnado), tope 30 (31 = no llegó); 9 consultas | Con el delito elegido, mediana 22 frente a 31 del portal; elegir el delito ahorra 7.0 [+1.8, +13.2] |
| **Coherencia de familias** (`familias --coherencia`) | ¿Los tipos agrupados se parecen entre sí? | Tipos (≥ 3 sentencias) cuyo centroide queda más cerca de su familia que de otra | 166 de 246 |
| **Latencia p50 / p95** (`latencia`) | ¿Cuánto espera el usuario? | Mediana y percentil 95 por consulta | 30.8 s / 52.6 s, 5 llamadas al LLM (p50) |

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
