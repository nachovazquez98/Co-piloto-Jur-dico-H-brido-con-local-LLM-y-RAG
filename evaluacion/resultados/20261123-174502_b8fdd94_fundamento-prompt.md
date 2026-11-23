# Evaluación: `fundamento-prompt`

- Fecha: 2026-11-23T17:45:02
- Commit: `b8fdd94`
- Huella del corpus: `77466c884916dd0e` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `a15b83c852281010`
- Parámetros: `{"n": 120, "particion": "desarrollo", "muestra": "primeras n por sha256(doc_id)"}`

> ⚠️ Distinto de lo congelado: codigo, prompt sintesis.

## Fundamento legal: ficha precalculada frente al prompt con el fundamento de la Sala

Mismas sentencias de desarrollo y misma referencia (artículos que la Sala invoca como fundamento, leídos por reglas). Diferencia pareada (prompt nuevo − ficha precalculada) con IC 95 % por bootstrap; se acepta la mejora solo si el intervalo no incluye el cero.

| medida | fichas | ficha precalculada | prompt con fundamento de la Sala | diferencia | significativa | efecto mínimo detectable |
|---|---|---|---|---|---|---|
| precisión | 120 | 0.719 | 0.729 | +0.010 [-0.040, +0.060] | no | 0.071 |
| exhaustividad (recall) | 120 | 0.146 | 0.123 | -0.023 [-0.041, -0.005] | sí | 0.026 |

## Artículos citados por ficha

Promedio por ficha válida.

| fichas generadas | JSON válido | sin artículos con el prompt nuevo | citados antes | citados después |
|---|---|---|---|---|
| 120 | 120 | 0 | 3.440 | 2.880 |

## Ficha precalculada: artículos citados fuera del fundamento de la Sala

Si aparecen en otra parte de la sentencia (agravios, primera instancia), el error es de selección y no una alucinación.

| citados fuera del fundamento | aparecen en la sentencia |
|---|---|
| 127 | 126 |
