# Evaluación: `fundamento-contexto`

- Fecha: 2026-11-23T21:24:26
- Commit: `5fc9b67`
- Huella del corpus: `77466c884916dd0e` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `a15b83c852281010`
- Parámetros: `{"n": 120, "particion": "desarrollo", "variante": "contexto", "muestra": "primeras n por sha256(doc_id)"}`

> ⚠️ Distinto de lo congelado: codigo, configuracion, prompt sintesis.

## Fundamento legal: ficha precalculada frente al contexto con la parte considerativa

Mismas sentencias de desarrollo y misma referencia (artículos que la Sala invoca como fundamento, leídos por reglas). Diferencia pareada (prompt nuevo − ficha precalculada) con IC 95 % por bootstrap; se acepta la mejora solo si el intervalo no incluye el cero.

| medida | fichas | ficha precalculada | contexto con la parte considerativa | diferencia | significativa | efecto mínimo detectable |
|---|---|---|---|---|---|---|
| precisión | 119 | 0.717 | 0.669 | -0.048 [-0.096, +0.006] | no | 0.071 |
| exhaustividad (recall) | 119 | 0.147 | 0.116 | -0.030 [-0.051, -0.012] | sí | 0.028 |

## Artículos citados por ficha

Promedio por ficha válida.

| fichas generadas | JSON válido | sin artículos con la variante | citados antes | citados después |
|---|---|---|---|---|
| 120 | 119 | 0 | 3.450 | 3.080 |

## Ficha precalculada: artículos citados fuera del fundamento de la Sala

Si aparecen en otra parte de la sentencia (agravios, primera instancia), el error es de selección y no una alucinación.

| citados fuera del fundamento | aparecen en la sentencia |
|---|---|
| 127 | 126 |

## Controles: lo que la variante no debe empeorar

Sentido del fallo frente al leído por reglas en los puntos resolutivos (sentencias donde las reglas lo leen) y número de agravios extraídos. Diferencia pareada con IC 95 % por bootstrap.

| control | fichas | ficha precalculada | contexto con la parte considerativa | diferencia | significativa |
|---|---|---|---|---|---|
| sentido igual al de los resolutivos | 96 | 0.854 | 0.812 | -0.042 [-0.104, +0.021] | no |
| agravios por ficha | 119 | 1.185 | 1.042 | -0.143 [-0.345, +0.050] | no |
