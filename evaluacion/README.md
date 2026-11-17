# Evaluación

Todos los comandos: `python -m copiloto_juridico.evaluacion <comando>` desde la raíz del repositorio.

## Evaluación automática, sin anotadores (v11.2)

No hay abogados disponibles para anotar, así que la referencia se construye sin intervención humana:

| Qué haría el abogado | Referencia automática |
|---|---|
| Revisar las consultas | Comprobaciones de la guía (fuga, nombre del tipo, datos identificables, negativas del modelo) y aceptación de los dos jueces LLM (`referencias`) |
| Juzgar la relevancia (0/1/2) | Mismo delito: familia de la **categoría del STJ** (externa). Mismo problema procesal: mayoría entre las reglas y los dos jueces. 2 = ambos, 1 = uno, 0 = ninguno |
| Anotar fichas | Familia: categoría del STJ. Acto impugnado, apelante y sentido: mayoría entre reglas y jueces (sobre todas las sentencias) |
| Kappa entre anotadores | Kappa entre reglas y cada juez, por campo y en la relevancia |
| Revisar las familias | `familias --coherencia`: centroides de embeddings por tipo y familia |
| Estudio de uso | `esfuerzo`: sentencias que hay que abrir para encontrar 3 útiles (portal, búsqueda por palabras, app) |

Los jueces (`config.MODELOS_JUECES`: `qwen2.5:14b` y `gemma2:9b`) son modelos de familias distintas a la de la app y
corren **uno a la vez** (8 GB de VRAM; `qwen2.5:14b` usa además el procesador). Un juez solo cuenta si acierta la familia
de la categoría del STJ en al menos `JUEZ_EXACTITUD_MINIMA` = 0.80 de las sentencias, criterio fijado antes de
correrlos. Límite que hay que declarar: es una evaluación contra lo que dicen el tribunal y el texto, no contra el
juicio de un experto sobre la utilidad de un precedente.

Orden, con el corpus definitivo (después de las descargas):

```bash
copiloto-datos construir-corpus --pdfs CARPETA ...  # y luego: copiloto-datos indexar (incluye metadatos)
fichas-batch                     # fichas de las sentencias nuevas o cambiadas (llama3.1:8b)
borradores --ampliar 10          # consultas nuevas hasta 10 por familia (llama3.1:8b)
conceptos                        # conceptos clave de las consultas (llama3.1:8b)
jueces --pares                   # pares (consulta, sentencia) a juzgar; sin LLM
jueces --modelo qwen2.5:14b      # una tanda por juez (reanudable)
jueces --modelo gemma2:9b
referencias                      # referencias, juicios y consultas validadas; sin LLM
modelos-tareas                   # qué modelo rinde mejor en conceptos, QC_1 y fichas (desarrollo)
congelar
recuperacion --particion prueba
qc --particion prueba
fichas-referencia --particion prueba
esfuerzo --particion prueba
familias --coherencia
```

Las consultas validadas no necesitan `revisado: true`; las demás no entran a la evaluación. Si algún día hay un
abogado, las secciones siguientes siguen valiendo y sus hojas sustituyen a la referencia automática
(`COPILOTO_EVALUACION_AUTOMATICA=false`).

## Evaluación con anotadores humanos (si hay un abogado disponible)

Ningún número es reportable mientras haya consultas con `revisado: false` (y sin validar): la partición de
**prueba** se rechaza hasta entonces.

### 1. Abogado penalista: revisar las familias de delito — `familias_penal.csv`

Cada tipo penal del portal está asignado a una familia. Revisa la columna `familia` (sobre todo las filas con
`revisar = si`, que traen la duda en `nota`), corrígela si hace falta y pon `revisado = si`. El sistema usa este
archivo tal como quede.

### 2. Revisar las consultas — `consultas_penal.jsonl`

Son borradores escritos por el LLM a partir de cada sentencia (comando `borradores`). Para cada línea:

- Léela como defensor: ¿así describirías tu caso? Corrige lo que haga falta.
- No debe copiar frases de la sentencia, nombrar el delito tal como lo etiqueta el portal ni traer fechas, años,
  cifras o duraciones de pena que identifiquen la sentencia (`validar` lo comprueba). Las líneas con
  "ATENCIÓN" en `notas` tienen alguno de estos problemas.
- Cambia `"revisado": false` por `"revisado": true`.

Las líneas `S…` (`sin_respuesta: true`) son temas sin sentencias en el corpus; sirven para medir si el sistema lo
reconoce. Al terminar: `validar`.

### 2b. Consultas reales — `consultas_reales.jsonl`

Al menos 20 consultas escritas por abogados penalistas **sin ver sentencias del corpus**, tal como las escribirían
en la app (el esquema está en las primeras líneas del archivo). Las consultas derivadas de sentencias facilitan la
búsqueda (comparten vocabulario con su sentencia); `recuperacion` reporta por separado derivadas y reales para medir
ese sesgo. Después de agregarlas, vuelve a correr `conceptos` y `hojas-anotacion`.

### 3a. Piloto de anotación — `piloto_juicios.csv` (primero)

10 consultas × 5 resultados que juzgan **dos personas por separado** (`relevancia_a`/`anotador_a` y
`relevancia_b`/`anotador_b`) con la guía de abajo. `kappa-piloto` calcula el acuerdo y lista los pares en desacuerdo.
Si kappa (0/1/2) es menor a 0.6, discutan los desacuerdos, precisen la guía en este archivo y repitan el piloto
**antes** de los 300 juicios.

### 3. Abogado: juzgar la relevancia — `anotacion_juicios.csv`

Una fila por par consulta–sentencia, en orden aleatorio (no muestra el puntaje del sistema). Lee la consulta y el
resumen de la sentencia (si hay duda, la sentencia completa) y escribe en `relevancia`:

| Grado | Criterio | Ejemplo para "robo de celular con violencia en vía pública" |
|---|---|---|
| **2** | Mismo delito y contexto compatible; útil como precedente directo. | Robo calificado con violencia a un transeúnte. |
| **1** | Misma familia de delito, o mismo problema procesal con hechos distintos. | Robo equiparado; robo a casa habitación. |
| **0** | Otro delito, o coincidencia solo de palabras. | Violencia familiar donde se menciona un celular. |

Opcional, pero útil para separar las dos cosas que mezcla el grado 1: `mismo_delito` (sí/no) y
`mismo_problema_procesal` (sí/no; mismo acto impugnado o mismo argumento). `recuperacion` reporta P@5 por cada una.

Escribe tu nombre en `anotador`. Una segunda persona juzga el 20 % de los pares en filas duplicadas con su propio
nombre; `validar` calcula el acuerdo (kappa de Cohen). Si es menor a 0.6, se revisa este criterio.

### 4. Abogado: anotar fichas — `anotacion_fichas.csv`

Para cada sentencia, leyéndola completa **antes de mirar la ficha del sistema**: `sentido_fallo` (CONFIRMA /
MODIFICA / REVOCA / REPONE / OTRO), `etapa`, `acto_impugnado`, `delito`, `articulos` (números separados por `;`) y
`agravios`. Después lee `ficha_app`, que es exactamente la ficha precalculada que muestra la app (`version_ficha` dice
cuál), y escribe `ficha_aceptable` (si/no) y, en `errores_ficha`, qué dato está mal. Las filas vacías se ignoran.

## Orden de los comandos

```bash
familias            # 1. tabla tipo -> familia (no sobrescribe una revisada)
plantilla           #    muestreo estratificado de sentencias (consultas_penal_borrador.jsonl)
borradores          # 2. el LLM redacta los borradores (consultas_penal.jsonl)
borradores --solo-atencion  # regenera los marcados ATENCIÓN con el prompt estricto
validar             #    fuga, nombres de tipo, acuerdo entre anotadores
conceptos           #    conceptos clave con el LLM, una vez
fichas-batch        #    ficha de cada sentencia penal, una vez (reanudable; ~8 h) + diagnóstico sin anotación
hojas-anotacion     # 3-4. piloto, hoja de juicios y hoja de fichas
kappa-piloto        #    acuerdo del piloto (si < 0.6, ajustar la guía antes de los 300 juicios)
recuperacion --particion desarrollo
ablacion --particion desarrollo
pesos               #    elige en desarrollo; mide en prueba una sola vez
qc --particion desarrollo
generacion
latencia --ejemplos --max 10
regresion           #    diagnóstico rápido con los ejemplos de la app (no es evaluación)
qc-diagnostico      #    tasa de descarte de QC_1 sin anotaciones (no es evaluación)
errores --corrida ID [ID ...]  # un caso por fallo con su tipo (evaluacion/errores/) y casos_presentacion.md
modelos             #    modelos de Ollama ya instalados: latencia, fichas sin anotación, VRAM (córrelo solo)
robustez            #    consultas difíciles con la app completa (diagnóstico)
estudio             #    estudio de uso manual vs. app (instrucciones en estudio_uso/README.md)
```

`errores` clasifica cada fallo en el primer paso del pipeline que falla: `cobertura` (no hay sentencias de ese tipo),
`clasificacion` (la familia correcta quedó fuera del top 3), `orden` (la familia estaba, pero el precedente relevante
quedó fuera del top k), `qc1_descarte` (QC_1 eliminó un candidato relevante), `ficha` (sentido, delito o artículos
no coinciden) u `otro`. Sin juicios humanos, "relevante" es "de la familia correcta".

## Corrida final: congelar y medir una sola vez

Cuando la configuración está elegida en desarrollo y las anotaciones están completas:

```bash
congelar                                  # evaluacion/congelado.json: código, corpus, índice, fichas, configuración, prompts, digest del modelo
recuperacion --particion prueba
qc --particion prueba
generacion                                # la hoja de fichas no tiene particiones: siempre es evaluación final
```

Cada corrida final se niega a empezar si algo no coincide con `congelado.json` (lo dice en una línea) y solo se
permite **una vez** por comando y congelado (`prueba_corridas.jsonl` guarda la marca). Repetirla exige
`--forzar "motivo"`, y el motivo queda en el reporte. Las corridas de desarrollo solo avisan si algo cambió.
