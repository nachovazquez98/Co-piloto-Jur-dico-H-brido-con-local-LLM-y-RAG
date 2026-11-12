# Evaluación del Co-piloto Jurídico (v11)

Datos y resultados de la evaluación de la sección 6 del informe. El código está en
`src/copiloto_juridico/evaluacion/`. Todos los comandos se ejecutan desde la raíz del repositorio:

```bash
python -m copiloto_juridico.evaluacion --help
```

> Ningún número de esta carpeta es reportable mientras haya consultas con `borrador: true`. Las corridas
> lo indican en su encabezado.

## Archivos

| Archivo | Quién lo llena | Contenido |
|---|---|---|
| `familias_penal.csv` | Generado; **lo revisa el abogado** | Tipo penal → familia de delito. Editar `familia` y poner `revisado = si`. Las filas con `revisar = si` tienen una nota con la duda. |
| `consultas_penal_borrador.jsonl` | Generado (plantilla) | 100 sentencias (5 por familia con ≥ 10 sentencias) + 20 de tipos raros + 10 temas sin respuesta, con partición 50/50 estratificada. Trae un extracto de cada sentencia. |
| `consultas_penal.jsonl` | **Persona** | Conjunto final (A, B y sin respuesta). Ver formato abajo. |
| `juicios.csv` | **Abogado** | Conjunto B: relevancia 0/1/2 de los 10 primeros resultados de cada consulta. |
| `fichas_anotadas.jsonl` | **Abogado** | Conjunto C: datos correctos de 40 sentencias. |
| `pendientes_penal.csv` | Generado | Tipos penales incompletos por prioridad, para descargar PDFs a mano. |
| `cache/conceptos.json` | Generado (LLM) | Conceptos clave de cada consulta (los experimentos sin LLM los leen de aquí). |
| `cache/llm.jsonl` | Generado (LLM) | Caché de respuestas del LLM de `qc` y `generacion` (no se versiona). |
| `resultados/` | Generado | Cada corrida: `<fecha>_<commit>_<comando>.json` y `.md`. |

### Formato de `consultas_penal.jsonl` (una consulta por línea)

```json
{"id": "A001", "consulta": "Mi defendido fue vinculado a proceso por quitarle el celular a una persona en la calle...",
 "doc_id_fuente": "2025-03-14-..._penal_robo-calificado.pdf", "tipo": "penal/robo-calificado", "familia": "robo",
 "particion": "desarrollo", "sin_respuesta": false, "borrador": false}
```

- **Conjunto A (documento conocido):** `doc_id_fuente` es la sentencia de la que se redactó la consulta.
- **Conjunto B (consultas reales):** `doc_id_fuente`, `tipo` y `familia` en `null`; la relevancia sale de `juicios.csv`.
- **Sin respuesta:** `sin_respuesta: true`, `doc_id_fuente: null`. Temas que no tienen tipo en el corpus.

**Reglas al redactar una consulta del conjunto A:** escríbela como la escribiría un defensor (hechos y problema
procesal: qué se apela y por qué), con tus palabras. No copies frases de la sentencia: `validar` rechaza
cualquier secuencia de más de 6 palabras compartida. Tampoco uses el nombre técnico del tipo ni el resultado
de la apelación.

## Cómo anotar la relevancia (conjunto B, escala del Anexo B)

| Grado | Criterio | Ejemplo para "robo de celular con violencia en vía pública" |
|---|---|---|
| **2** | Mismo delito y contexto fáctico compatible; útil como precedente directo. | Apelación de sentencia por robo calificado con violencia a transeúnte. |
| **1** | Misma familia de delito o mismo problema procesal con hechos distintos. | Robo equiparado; o robo calificado a casa habitación. |
| **0** | Otro delito, o coincidencia solo de palabras. | Violencia familiar en la que se menciona un celular. |

- Se juzga leyendo la ficha y, si hay duda, la sentencia original.
- El anotador **no ve** el puntaje del sistema ni el orden en que apareció el documento.
- Una segunda persona juzga el 20 % de los pares; `validar` calcula el kappa de Cohen. Si es < 0.6, se revisa la guía.

`juicios.csv`:

```csv
consulta_id,doc_id,relevancia,anotador
B001,2025-03-14-..._penal_robo-calificado.pdf,2,abogado1
```

## Formato de `fichas_anotadas.jsonl` (conjunto C)

```json
{"doc_id": "2025-..._penal_robo-calificado.pdf", "sentido_fallo": "CONFIRMA", "etapa": "sentencia definitiva de juicio oral",
 "delito": "robo calificado", "articulos": ["233", "236"], "agravios": "…",
 "consulta_id": "B004", "ficha_aceptable": true}
```

`sentido_fallo` ∈ CONFIRMA / MODIFICA / REVOCA / REPONE / OTRO. `consulta_id` y `ficha_aceptable` (¿la ficha
generada es correcta y útil para esa consulta?) son opcionales y sirven para evaluar QC_2.

## Orden de trabajo

```bash
# 0. (Opcional) Ampliar el corpus ANTES de redactar consultas y congelarlo después
python -m copiloto_juridico.cli pendientes --salida evaluacion/pendientes_penal.csv

# 1. Familias de delito -> revisión del abogado
python -m copiloto_juridico.evaluacion familias

# 2. Plantilla del conjunto A (borrador). --redactar-con-llm propone textos que hay que revisar
python -m copiloto_juridico.evaluacion plantilla
#    redactar/corregir, poner borrador:false, guardar como consultas_penal.jsonl
python -m copiloto_juridico.evaluacion validar

# 3. Conceptos clave (LLM, una vez)
python -m copiloto_juridico.evaluacion conceptos

# 4. Experimentos sin LLM (minutos) en desarrollo
python -m copiloto_juridico.evaluacion recuperacion --particion desarrollo
python -m copiloto_juridico.evaluacion ablacion --particion desarrollo
python -m copiloto_juridico.evaluacion pesos             # elige en desarrollo; prueba una sola vez

# 5. Con LLM (horas; se pueden interrumpir y reanudar gracias a la caché)
python -m copiloto_juridico.evaluacion qc --particion todas
python -m copiloto_juridico.evaluacion generacion --prompt-penal v11
python -m copiloto_juridico.evaluacion latencia --particion prueba --max 10

# 6. Reporte final: repetir 4 y 5 con --particion prueba UNA vez, con la configuración elegida
```

Para comparar antes/después, cada cambio de v11 es una variable de entorno (ver `config.py`), por ejemplo
`COPILOTO_QC1_CONTEXTO=fragmento_similar python -m copiloto_juridico.evaluacion qc`.
