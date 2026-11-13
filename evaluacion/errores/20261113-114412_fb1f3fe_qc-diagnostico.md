# Errores de `20261113-114412_fb1f3fe_qc-diagnostico`

- Comando: `qc-diagnostico` · commit `fb1f3fe` · corpus `bea10929ce634a58`
- Diagnóstico (consultas de ejemplo, no evaluación).
- Relevante = de la familia correcta (sin juicios humanos).
- Fichas: precalculadas (`data/fichas/fichas-penal.parquet`); las marcas son las deterministas de la app.

## Conteo por tipo

| tipo | fallos | qué es |
|---|---|---|
| cobertura | 0 | no hay sentencias de ese tipo en el corpus |
| clasificacion | 0 | la familia correcta quedó fuera del top 3 |
| orden | 0 | la familia estaba, pero el precedente relevante quedó fuera del top k |
| qc1_descarte | 6 | QC_1 eliminó un candidato relevante |
| ficha | 0 | el sentido del fallo, el delito o los artículos de la ficha no coinciden |
| otro | 0 | la corrida marca un fallo que no encaja en los tipos anteriores |

## 1. Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral — `qc1_descarte` (también: orden, ficha)

> Mi representado disparó contra otra persona durante una discusión y la lesionó en una pierna; el Ministerio Público lo acusa de tentativa de homicidio calificado. La defensa sostiene que solo hubo lesiones porque no se acreditó la intención de matar. ¿Cómo ha resuelto la Sala la reclasificación en apelación?

- Qué pasó: QC_1 descartó el toca 391/2024 (homicidio, vinculación), que sí es de la familia de la consulta.
- Familia esperada: homicidio (163 sentencias en el corpus)
- Top 3 de familias: homicidio (0.633) · lesiones (0.276) · feminicidio (0.250)
- Acto impugnado de la consulta: no lo menciona
- Nota: QC_1 se evaluó sobre los 5 primeros candidatos; se muestran los 3 primeros.

| # | toca | familia | acto impugnado | relevante | QC_1 | ficha: verificar |
|---|---|---|---|---|---|---|
| 1 | 53/2025 | lesiones | sentencia definitiva | no | SI | artículos no encontrados en la sentencia: 219 |
| 2 | 54/2025 | homicidio | sentencia definitiva | sí | SI | — |
| 3 | 391/2024 | homicidio | vinculación | sí | NO | — |

## 2. Penal · Abuso sexual infantil · Sentencia de juicio oral — `qc1_descarte`

> Se condenó a mi defendido por tocamientos a una niña de 9 años. La defensa alega contradicciones entre la entrevista inicial de la menor y su declaración en juicio, y que la perito no siguió el protocolo. Busco cómo valora la Sala el testimonio de menores y los agravios de la defensa en apelación.

- Qué pasó: QC_1 descartó el toca 286/2024 (delitos sexuales, sentencia definitiva), que sí es de la familia de la consulta.
- Familia esperada: delitos sexuales (148 sentencias en el corpus)
- Top 3 de familias: delitos sexuales (0.547) · trata de personas (0.241) · contra la dignidad de las personas (0.235)
- Acto impugnado de la consulta: sentencia definitiva
- Nota: QC_1 se evaluó sobre los 5 primeros candidatos; se muestran los 3 primeros.

| # | toca | familia | acto impugnado | relevante | QC_1 | ficha: verificar |
|---|---|---|---|---|---|---|
| 1 | 192/2023 | delitos sexuales | sentencia definitiva | sí | SI | — |
| 2 | 277/2023 | delitos sexuales | sentencia definitiva | sí | SI | — |
| 3 | 141/2025 | delitos sexuales | sentencia definitiva | sí | SI | — |

Descartes de QC_1 fuera de los primeros: toca 286/2024 (delitos sexuales, relevante: sí)

## 3. Penal · Violación · Vinculación a proceso — `qc1_descarte` (también: ficha)

> Mi cliente fue vinculado a proceso por violación; la denunciante y él tenían una relación previa y hay mensajes posteriores al hecho. La defensa apela la vinculación porque el dato de prueba médico no es concluyente. ¿Qué criterios ha seguido la Sala?

- Qué pasó: QC_1 descartó el toca 199/2025 (delitos sexuales, vinculación), que sí es de la familia de la consulta.
- Familia esperada: delitos sexuales (148 sentencias en el corpus)
- Top 3 de familias: delitos sexuales (0.585) · violencia familiar (0.352) · responsabilidad profesional (0.266)
- Acto impugnado de la consulta: vinculación
- Nota: QC_1 se evaluó sobre los 5 primeros candidatos; se muestran los 3 primeros.

| # | toca | familia | acto impugnado | relevante | QC_1 | ficha: verificar |
|---|---|---|---|---|---|---|
| 1 | 199/2025 | delitos sexuales | vinculación | sí | NO | artículos no encontrados en la sentencia: 316 |
| 2 | [2025-08-26-120921] | delitos sexuales | vinculación | sí | SI | — |
| 3 | 159/2025 | delitos sexuales | vinculación | sí | SI | — |

## 4. Penal · Fraude · No vinculación (apela el MP) — `qc1_descarte`

> El Ministerio Público apeló el auto de no vinculación a proceso dictado a favor de mi cliente, acusado de fraude por no entregar un terreno que vendió. La defensa sostiene que es un incumplimiento civil de contrato y no un engaño. Necesito precedentes para contestar los agravios del Ministerio Público.

- Qué pasó: QC_1 descartó el toca 17/2025 (fraude y abuso de confianza, no vinculación), que sí es de la familia de la consulta.
- Familia esperada: fraude y abuso de confianza (132 sentencias en el corpus)
- Top 3 de familias: fraude y abuso de confianza (0.659) · falsedad y falsificacion (0.192) · otros (0.189)
- Acto impugnado de la consulta: no vinculación
- Nota: QC_1 se evaluó sobre los 5 primeros candidatos; se muestran los 3 primeros.

| # | toca | familia | acto impugnado | relevante | QC_1 | ficha: verificar |
|---|---|---|---|---|---|---|
| 1 | 85/2025 | fraude y abuso de confianza | no vinculación | sí | SI | — |
| 2 | 356/2025 | fraude y abuso de confianza | no vinculación | sí | SI | — |
| 3 | [2025-04-03-120507] | fraude y abuso de confianza | no vinculación | sí | SI | — |

Descartes de QC_1 fuera de los primeros: toca 17/2025 (fraude y abuso de confianza, relevante: sí)

## 5. Penal · Abandono de familiares · Sentencia de juicio oral — `qc1_descarte`

> Condenaron a mi cliente por dejar de pagar la pensión de sus hijos durante un año. Él perdió su empleo y tiene constancias de que buscó trabajo. Busco precedentes en los que la Sala haya valorado la imposibilidad económica como causa para absolver o modificar la pena.

- Qué pasó: QC_1 descartó el toca 197/2025 (abandono de familiares, otro), que sí es de la familia de la consulta.
- Familia esperada: abandono de familiares (33 sentencias en el corpus)
- Top 3 de familias: abandono de familiares (0.520) · contra la dignidad de las personas (0.235) · otros (0.192)
- Acto impugnado de la consulta: no lo menciona
- Nota: QC_1 se evaluó sobre los 5 primeros candidatos; se muestran los 3 primeros.

| # | toca | familia | acto impugnado | relevante | QC_1 | ficha: verificar |
|---|---|---|---|---|---|---|
| 1 | 25/2024 | abandono de familiares | sentencia definitiva | sí | SI | — |
| 2 | 87/2025 | abandono de familiares | no vinculación | sí | SI | — |
| 3 | 131/2025 | abandono de familiares | vinculación | sí | SI | — |

Descartes de QC_1 fuera de los primeros: toca 197/2025 (abandono de familiares, relevante: sí)

## 6. Penal · Secuestro agravado · Sentencia de juicio oral — `qc1_descarte` (también: orden)

> A mi defendido se le acusa de participar en un secuestro como quien cuidaba a la víctima en una casa de seguridad. La defensa sostiene que no se acreditó su intervención y que el reconocimiento fue inducido. Necesito precedentes sobre la apelación de sentencias condenatorias por secuestro.

- Qué pasó: QC_1 descartó el toca 448/2024 (privacion de la libertad y secuestro, vinculación), que sí es de la familia de la consulta.
- Familia esperada: privacion de la libertad y secuestro (80 sentencias en el corpus)
- Top 3 de familias: privacion de la libertad y secuestro (0.629) · delitos sexuales (0.341) · contra la dignidad de las personas (0.232)
- Acto impugnado de la consulta: no lo menciona
- Nota: QC_1 se evaluó sobre los 5 primeros candidatos; se muestran los 3 primeros.

| # | toca | familia | acto impugnado | relevante | QC_1 | ficha: verificar |
|---|---|---|---|---|---|---|
| 1 | 266/2025 | privacion de la libertad y secuestro | no determinado | sí | SI | — |
| 2 | 448/2024 | privacion de la libertad y secuestro | vinculación | sí | NO | — |
| 3 | [2025-08-29-152021] | delitos sexuales | vinculación | no | NO | — |

Descartes de QC_1 fuera de los primeros: toca 427/2024 (delitos sexuales, relevante: no)
