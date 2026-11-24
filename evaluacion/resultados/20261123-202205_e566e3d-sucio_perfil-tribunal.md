# Evaluación: `perfil-tribunal`

- Fecha: 2026-11-23T20:22:05
- Commit: `e566e3d` (con cambios sin confirmar)
- Huella del corpus: `77466c884916dd0e` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `a15b83c852281010`
- Parámetros: `{"particion": "desarrollo", "prompt": "PROMPT_CONSULTA_TRIBUNAL"}`

> ⚠️ Distinto de lo congelado: codigo, configuracion, prompt sintesis.
> ⚠️ Consultas validadas automáticamente (comprobaciones de la guía + jueces LLM): 42; descartadas: 63 (A004, A006, A012, A016, A022, A026, A029, A031, A035, A037...).

## Perfil del tribunal de alzada frente al del abogado

Mismas consultas validadas, reescritas por el LLM como toca en estudio; misma referencia automática. Hit@3 de familia sin la sentencia fuente (leave-one-out); nDCG@5 de la lista de la app (filtro top-3, RRF k=60). Diferencia pareada con IC 95 % por bootstrap.

| métrica | consultas | abogado | tribunal de alzada | diferencia (tribunal − abogado) | significativa |
|---|---|---|---|---|---|
| Hit@3 de familia | 39 | 0.667 | 0.718 | +0.051 [-0.077, +0.205] | no |
| nDCG@5 contra la referencia | 39 | 0.266 | 0.145 | -0.121 [-0.205, -0.042] | sí |

## Ejemplos de reescritura

Las tres primeras consultas, en ambas redacciones.

| id | abogado | tribunal de alzada |
|---|---|---|
| A001 | Represento a un hombre que se le atribuye haber dejado sin cuidados a dos menores de edad. Se le acusa de no haber cumplido con sus obligaciones como padre y de no haber proporcionado el apoyo necesario a los niños. Quiero apelar la resolución que desestimó su planteamiento de incompetencia para conocer del caso, argumentando que el tribunal inferior no tenía jurisdicción sobre la materia. | Toca en estudio: Se trata de un caso penal en el que se le atribuye a un hombre haber dejado sin cuidados a dos menores de edad. El acusado fue desestimado por incompetencia para conocer del caso, y ahora apela argumentando que el tribunal inferior no tenía jurisdicción sobre la materia. Se busca determinar si la resolución apelada se ajusta a derecho en este sentido. |
| A003 | Represento a un joven acusado de haber dejado sin cuidados a sus hijos menores de edad y de haber resistido a las autoridades cuando se le ordenó comparecer. Mi cliente apeló la resolución que ordenó su sobreseimiento por falta de pruebas, pero ahora quiero saber si hay algún precedente en el que se haya anulado una decisión similar por considerarla injusta o contraria a derecho. | Toca en estudio: Un joven acusado de dejar sin cuidados a sus hijos menores de edad y resistir a las autoridades, apeló la resolución que ordenó su sobreseimiento por falta de pruebas. La decisión apelada fue emitida por un tribunal inferior y el joven sostiene que es injusta y contraria al derecho. Se busca determinar si hay algún precedente en el que se haya anulado una decisión similar. |
| A010 | Represento a un imputado acusado de haber ejercido presión sobre dos víctimas para que realizaran una acción en su beneficio. Mi cliente se encuentra en proceso de juicio y ha sido objeto de una resolución que considero injusta, por lo que deseo apelarla con el fin de revisar la legalidad de las actuaciones realizadas durante el trámite del caso. | Toca en estudio: Caso penal en que se investiga presunta coacción a víctimas para realizar una acción en beneficio del imputado. La resolución apelada es injusta, según la opinión de mi cliente, quien fue condenado por ejercicio de presión sobre dos víctimas. El imputado apela la decisión del tribunal inferior. |
