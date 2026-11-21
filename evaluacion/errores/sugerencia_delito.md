# Sugerencia inicial de delito: ¿qué tan seguido acierta? (v11.2)

La app sugiere, sin llamar al LLM, las familias de delito que deduce de la descripción (`sugerir_familias`): la
consulta hace de conceptos clave en el clasificador híbrido y se toman las familias de las 3 primeras categorías.
El usuario puede usarla, cambiarla o no elegir nada. Diagnóstico sin LLM ni partición de prueba, corpus penal
de 2,495 sentencias (c27877e).

**Casos:** las 100 consultas penales de **desarrollo** con sentencia fuente (sin esa sentencia en la búsqueda, para
que no vote por su propio tipo) y los 14 ejemplos de la app que tienen familia.

| Variante | Desarrollo: familia correcta sugerida (de 100) | …y en primer lugar | Ejemplos (de 14) |
|---|---|---|---|
| **Actual:** clasificador con la consulta como conceptos | **73** | 41 | **13** |
| Clasificador sin conceptos | 73 | 41 | 13 |
| Voto de las 10 primeras sentencias recuperadas | 63 | 46 | 12 |
| Voto de las 30 primeras sentencias recuperadas | 66 | 43 | 14 |
| Comparación: la app con los conceptos del LLM | 81 | 43 | — |

**Decisión:** se queda la variante actual. Es la mejor sin LLM en desarrollo, que es el conjunto grande; el voto
de 30 acierta un ejemplo más pero pierde 7 consultas de desarrollo. Queda 8 puntos por debajo de la clasificación
con el LLM, así que es una sugerencia y no un filtro automático. Si el usuario la acepta y no coincide con lo que
la app habría buscado, la nota de la app se lo dice, y la búsqueda sin filtro sigue completando los candidatos.

**Fallo visible:** el primer ejemplo de la app (robo con cuchillo y reconocimiento por fotografía) recibe "delitos
sexuales, feminicidio, homicidio". La app con el LLM lo clasifica como robo.
