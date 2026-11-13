# Casos de error para la presentación

Elegidos automáticamente por `errores` (uno por tipo de error presente). Diagnóstico con las consultas de ejemplo: no son tasas de error del sistema.

| tipo | consulta | qué pasó | corrida |
|---|---|---|---|
| qc1_descarte | Mi representado disparó contra otra persona durante una discusión y la lesionó en una pierna; el Ministerio Público… | QC_1 descartó el toca 391/2024 (homicidio, vinculación), que sí es de la familia de la consulta. | 20261113-114412_fb1f3fe_qc-diagnostico |
| ficha | Mi defendido fue vinculado a proceso porque supuestamente le arrebató el celular a una mujer en una parada… | Ficha del toca 416/2024: artículos no encontrados en la sentencia: 133, 47, 474. | 20261113-114442_fb1f3fe_regresion |
| orden | Mi cliente lesionó a un vecino con un tubo durante una pelea iniciada por el vecino; se le… | lesiones quedó en el top 3 de familias, pero solo 1 de los 3 primeros precedentes es de esa familia (homicidio · lesiones · homicidio). | 20261113-114442_fb1f3fe_regresion |
| cobertura | Mi defendido tiene 16 años y se le impuso una medida de internamiento por robo con violencia. La… | No hay sentencias de ese tipo en el corpus; los precedentes más cercanos son de robo · robo · robo. | 20261113-114442_fb1f3fe_regresion |

Corregidos en v11.1 por el orden por familia + acto impugnado (antes eran fallos de `orden`):

| caso | antes | después | de la familia correcta |
|---|---|---|---|
| Penal · Robo calificado · Vinculación a proceso | homicidio (sentencia definitiva) · delitos sexuales (otro) · robo (otro) | robo (vinculación) · robo (vinculación) · robo (vinculación) | 1/3 → 3/3 |
| Penal · Homicidio calificado · Prisión preventiva | robo (medida cautelar) · robo (medida cautelar) · homicidio (medida cautelar) | homicidio (medida cautelar) · homicidio (medida cautelar) · homicidio (medida cautelar) | 1/3 → 3/3 |
| Penal · Violación · Vinculación a proceso | violencia familiar (no vinculación) · delitos sexuales (vinculación) · delitos sexuales (no determinado) | delitos sexuales (vinculación) · delitos sexuales (vinculación) · delitos sexuales (vinculación) | 2/3 → 3/3 |
