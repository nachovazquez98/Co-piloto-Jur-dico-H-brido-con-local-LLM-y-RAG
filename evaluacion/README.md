# Evaluación: qué tiene que hacer cada persona

Ningún número es reportable mientras haya consultas con `revisado: false`: la partición de **prueba** se rechaza
hasta entonces. Todos los comandos: `python -m copiloto_juridico.evaluacion <comando>` desde la raíz del repositorio.

## 1. Abogado penalista: revisar las familias de delito — `familias_penal.csv`

Cada tipo penal del portal está asignado a una familia. Revisa la columna `familia` (sobre todo las filas con
`revisar = si`, que traen la duda en `nota`), corrígela si hace falta y pon `revisado = si`. El sistema usa este
archivo tal como quede.

## 2. Revisar las consultas — `consultas_penal.jsonl`

Son borradores escritos por el LLM a partir de cada sentencia (comando `borradores`). Para cada línea:

- Léela como defensor: ¿así describirías tu caso? Corrige lo que haga falta.
- No debe copiar frases de la sentencia ni nombrar el delito tal como lo etiqueta el portal (`validar` lo comprueba).
- Cambia `"revisado": false` por `"revisado": true`.

Las líneas `S…` (`sin_respuesta: true`) son temas sin sentencias en el corpus; sirven para medir si el sistema lo
reconoce. Al terminar: `validar`.

## 3. Abogado: juzgar la relevancia — `anotacion_juicios.csv`

Una fila por par consulta–sentencia, en orden aleatorio (no muestra el puntaje del sistema). Lee la consulta y el
resumen de la sentencia (si hay duda, la sentencia completa) y escribe en `relevancia`:

| Grado | Criterio | Ejemplo para "robo de celular con violencia en vía pública" |
|---|---|---|
| **2** | Mismo delito y contexto compatible; útil como precedente directo. | Robo calificado con violencia a un transeúnte. |
| **1** | Misma familia de delito, o mismo problema procesal con hechos distintos. | Robo equiparado; robo a casa habitación. |
| **0** | Otro delito, o coincidencia solo de palabras. | Violencia familiar donde se menciona un celular. |

Escribe tu nombre en `anotador`. Una segunda persona juzga el 20 % de los pares en filas duplicadas con su propio
nombre; `validar` calcula el acuerdo (kappa de Cohen). Si es menor a 0.6, se revisa este criterio.

## 4. Abogado: anotar fichas — `anotacion_fichas.csv`

Para cada sentencia, leyéndola completa: `sentido_fallo` (CONFIRMA / MODIFICA / REVOCA / REPONE / OTRO), `etapa`,
`acto_impugnado`, `delito`, `articulos` (números separados por `;`) y `agravios`. Las filas vacías se ignoran.

## Orden de los comandos

```bash
familias            # 1. tabla tipo -> familia (no sobrescribe una revisada)
plantilla           #    muestreo estratificado de sentencias (consultas_penal_borrador.jsonl)
borradores          # 2. el LLM redacta los borradores (consultas_penal.jsonl)
validar             #    fuga, nombres de tipo, acuerdo entre anotadores
conceptos           #    conceptos clave con el LLM, una vez
hojas-anotacion     # 3-4. hojas para el abogado
recuperacion --particion desarrollo
ablacion --particion desarrollo
pesos               #    elige en desarrollo; mide en prueba una sola vez
qc --particion todas
generacion
latencia --ejemplos --max 10
regresion           #    diagnóstico rápido con los ejemplos de la app (no es evaluación)
qc-diagnostico      #    tasa de descarte de QC_1 sin anotaciones (no es evaluación)
```

Para el número final se repiten `recuperacion`, `qc` y `generacion` con `--particion prueba`, **una sola vez**, con la
configuración elegida en desarrollo.
