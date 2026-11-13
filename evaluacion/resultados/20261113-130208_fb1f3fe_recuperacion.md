# Evaluación: `recuperacion`

> **BORRADOR — no reportable.** Hay consultas marcadas `borrador: true`.

- Fecha: 2026-11-13T13:02:08
- Commit: `fb1f3fe`
- Huella del corpus: `bea10929ce634a58` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `sin congelar`
- Parámetros: `{"particion": "desarrollo"}`

> ⚠️ Sin congelado: corrida de desarrollo no atada a una versión.
> ⚠️ Hay consultas en borrador: esta corrida NO es reportable.
> ⚠️ 3 consultas comparten más de 6 palabras seguidas con su sentencia fuente (fuga): A053, A077, A090. Corre `validar` para verlas.
> ⚠️ Sin juicios de relevancia anotados: nDCG@5 y P@5 humanos quedan por medir.

## Clasificación de la consulta

Hit@3 = la categoría correcta queda dentro del filtro de 3 categorías (si no, ninguna etapa posterior la recupera). La sentencia de la que se redactó la consulta no participa (leave-one-out). IC95 de Wilson; F1 macro sobre la predicción top-1. k-NN = tipo mayoritario de las 10 sentencias más cercanas.

| variante | n | Hit@3 tipo | Hit@3 tipo IC95 | Hit@3 familia | Hit@3 familia IC95 | Top-1 tipo | Top-1 tipo IC95 | Top-1 familia | Top-1 familia IC95 | F1 macro tipo | F1 macro familia |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Híbrido (app) | 60 | 0.783 | [0.664, 0.869] | 0.783 | [0.664, 0.869] | nan | — | 0.617 | [0.490, 0.729] | nan | 0.536 |
| k-NN 10 (línea base) | 60 | 0.133 | [0.069, 0.242] | 0.483 | [0.362, 0.607] | 0.100 | [0.047, 0.201] | 0.333 | [0.227, 0.459] | 0.061 | 0.313 |

## Diagnóstico: clasificación con la sentencia fuente incluida

Solo diagnóstico (no es la métrica): cuánto se inflaría la exactitud si la sentencia fuente votara por su propio tipo.

| variante | n | Hit@3 tipo | Hit@3 tipo IC95 | Hit@3 familia | Hit@3 familia IC95 | Top-1 tipo | Top-1 tipo IC95 | Top-1 familia | Top-1 familia IC95 | F1 macro tipo | F1 macro familia |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Híbrido (app), con fuente | 60 | 0.867 | [0.758, 0.931] | 0.867 | [0.758, 0.931] | nan | — | 0.633 | [0.507, 0.744] | nan | 0.549 |
| k-NN 10, con fuente | 60 | 0.300 | [0.199, 0.425] | 0.567 | [0.441, 0.684] | 0.167 | [0.093, 0.280] | 0.383 | [0.271, 0.510] | 0.107 | 0.369 |

## Errores de clasificación

Consultas cuya familia correcta quedó fuera del top-3.

| consulta | partición | tipo correcto | familia | top-3 |
|---|---|---|---|---|
| A010 | desarrollo | penal/chantaje | amenazas-y-extorsion | penal/familia/homicidio · penal/familia/privacion-de-la-libertad-y-secuestro · penal/familia/falsedad-y-falsificacion |
| A016 | desarrollo | penal/contra-la-dignidad-de-las-personas | contra-la-dignidad-de-las-personas | penal/familia/lesiones · penal/familia/delitos-sexuales · penal/familia/violencia-familiar |
| A020 | desarrollo | penal/contra-la-dignidad-de-las-personas | contra-la-dignidad-de-las-personas | penal/familia/lesiones · penal/familia/delitos-culposos · penal/familia/homicidio |
| A021 | desarrollo | penal/contra-la-salud-en-su-modalidad-de-narcomenudeo | contra-la-salud | penal/familia/fraude-y-abuso-de-confianza · penal/familia/robo · penal/familia/despojo |
| A023 | desarrollo | penal/narcomenudeo | contra-la-salud | penal/familia/robo · penal/familia/armas · penal/familia/otros |
| A029 | desarrollo | penal/dano-a-las-cosas | dano-en-las-cosas | penal/familia/violencia-familiar · penal/familia/lesiones · penal/familia/contra-la-dignidad-de-las-personas |
| A032 | desarrollo | penal/lesiones-a-titulo-de-culpa | delitos-culposos | penal/familia/lesiones · penal/familia/violencia-familiar · penal/familia/delitos-sexuales |
| A050 | desarrollo | penal/tentativa-de-despojo | despojo | penal/familia/robo · penal/familia/fraude-y-abuso-de-confianza · penal/familia/homicidio |
| A055 | desarrollo | penal/falsedad-en-declaraciones-y-en-informes-dados-a-una-autoridad | falsedad-y-falsificacion | penal/familia/fraude-y-abuso-de-confianza · penal/familia/violencia-familiar · penal/familia/homicidio |
| A057 | desarrollo | penal/feminicidio-en-grado-de-tentativa | feminicidio | penal/familia/delitos-sexuales · penal/familia/robo · penal/familia/homicidio |
| A058 | desarrollo | penal/tentativa-de-feminicidio | feminicidio | penal/familia/lesiones · penal/familia/homicidio · penal/familia/delitos-sexuales |
| A073 | desarrollo | penal/lesiones-dolosas | lesiones | penal/familia/homicidio · penal/familia/delitos-sexuales · penal/familia/violencia-familiar |
| A095 | desarrollo | penal/desvio-y-aprovechamiento-indebido-de-atribuciones-y-facultades | servidores-publicos-y-administracion-de-justicia | penal/familia/delitos-sexuales · penal/familia/trata-de-personas · penal/familia/fraude-y-abuso-de-confianza |

## Matriz de confusión por familia (top-1)

Filas = familia correcta, columnas = familia predicha. La diagonal son aciertos.

| verdadero | abandono-de-familiares | amenazas-y-extorsion | armas | contra-la-dignidad-de-las-personas | contra-la-salud | dano-en-las-cosas | delitos-culposos | delitos-sexuales | desaparicion-de-personas | despojo | falsedad-y-falsificacion | feminicidio | fraude-y-abuso-de-confianza | homicidio | lesiones | privacion-de-la-libertad-y-secuestro | responsabilidad-profesional | robo | servidores-publicos-y-administracion-de-justicia | violencia-familiar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abandono-de-familiares | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| amenazas-y-extorsion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| armas | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| contra-la-dignidad-de-las-personas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| contra-la-salud | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| dano-en-las-cosas | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| delitos-culposos | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| delitos-sexuales | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| desaparicion-de-personas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| despojo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| falsedad-y-falsificacion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| feminicidio | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| fraude-y-abuso-de-confianza | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| homicidio | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| lesiones | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| privacion-de-la-libertad-y-secuestro | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| responsabilidad-profesional | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| robo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| servidores-publicos-y-administracion-de-justicia | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| violencia-familiar | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 |

## Recuperación (conjunto A)

Hit@k = la sentencia de la que se redactó la consulta aparece en los k primeros; MRR = promedio de 1/rango (lista de hasta 30). nDCG@5 y P@5 por metadatos (2 = mismo tipo, 1 = misma familia) se calculan sin la sentencia fuente: miden si aparecen OTROS precedentes del mismo tipo; no sustituyen al juicio humano. IC95: Wilson para Hit, bootstrap (1,000) para MRR y nDCG.

| variante | n | Hit@1 | Hit@1 IC95 | Hit@5 | Hit@5 IC95 | Hit@10 | MRR | MRR IC95 | nDCG@5 (metadatos) | nDCG@5 (metadatos) IC95 | P@5 mismo tipo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Solo BM25 | 60 | 0.167 | [0.093, 0.280] | 0.300 | [0.199, 0.425] | 0.383 | 0.230 | [0.139, 0.323] | 0.312 | [0.247, 0.378] | 0.090 |
| Solo embeddings | 60 | 0.050 | [0.017, 0.137] | 0.217 | [0.131, 0.336] | 0.250 | 0.110 | [0.056, 0.178] | 0.142 | [0.100, 0.188] | 0.050 |
| RRF k=20, sin filtro | 60 | 0.167 | [0.093, 0.280] | 0.333 | [0.227, 0.459] | 0.433 | 0.248 | [0.160, 0.335] | 0.280 | [0.220, 0.342] | 0.087 |
| RRF k=60, sin filtro | 60 | 0.167 | [0.093, 0.280] | 0.333 | [0.227, 0.459] | 0.500 | 0.247 | [0.158, 0.334] | 0.274 | [0.212, 0.337] | 0.080 |
| RRF k=100, sin filtro | 60 | 0.167 | [0.093, 0.280] | 0.350 | [0.242, 0.476] | 0.500 | 0.244 | [0.157, 0.332] | 0.272 | [0.207, 0.335] | 0.073 |
| RRF k=60 + filtro top-1 | 60 | 0.133 | [0.069, 0.242] | 0.300 | [0.199, 0.425] | 0.383 | 0.210 | [0.130, 0.297] | 0.426 | [0.331, 0.518] | 0.120 |
| RRF k=60 + filtro top-3 (app) | 60 | 0.150 | [0.081, 0.261] | 0.350 | [0.242, 0.476] | 0.417 | 0.232 | [0.147, 0.319] | 0.342 | [0.274, 0.409] | 0.110 |
| RRF k=60 + filtro top-5 | 60 | 0.167 | [0.093, 0.280] | 0.367 | [0.256, 0.493] | 0.467 | 0.254 | [0.163, 0.342] | 0.320 | [0.253, 0.383] | 0.097 |
| RRF k=20 + filtro top-3 | 60 | 0.150 | [0.081, 0.261] | 0.350 | [0.242, 0.476] | 0.433 | 0.237 | [0.152, 0.323] | 0.340 | [0.274, 0.405] | 0.100 |
| RRF k=100 + filtro top-3 | 60 | 0.133 | [0.069, 0.242] | 0.317 | [0.213, 0.442] | 0.383 | 0.208 | [0.125, 0.293] | 0.336 | [0.267, 0.402] | 0.100 |
| RRF k=60 + filtro top-3 + re-ranker | 60 | 0.183 | [0.106, 0.299] | 0.367 | [0.256, 0.493] | 0.467 | 0.269 | [0.181, 0.359] | 0.368 | [0.299, 0.440] | 0.120 |

## Latencia del re-ranker (segundos por consulta)

Cross-encoder BAAI/bge-reranker-v2-m3 sobre hasta 30 candidatos; incluye la carga del modelo en la primera consulta.

| n | p50 | p95 | máximo |
|---|---|---|---|
| 120 | 1.491 | 1.582 | 6.882 |

## Comparaciones pareadas

Δ = media(A − B) sobre las mismas consultas, IC95 bootstrap pareado. Una mejora se acepta solo si el intervalo no incluye 0. Efecto mínimo detectable: la Δ más pequeña que esta n detectaría (alfa 5 %, potencia 80 %); una Δ menor no significativa no prueba que no haya diferencia.

| A | B | métrica | n | Δ | Δ IC95 | ¿significativa? | gana A / B / empate | efecto mínimo detectable |
|---|---|---|---|---|---|---|---|---|
| RRF k=60, sin filtro | Solo BM25 | RR (documento fuente) | 60 | 0.017 | [-0.056, +0.096] | no | 24 / 14 / 22 | ±0.110 |
| RRF k=60, sin filtro | Solo BM25 | nDCG@5 (metadatos) | 60 | -0.038 | [-0.088, +0.008] | no | 22 / 27 / 11 | ±0.070 |
| RRF k=60, sin filtro | Solo embeddings | RR (documento fuente) | 60 | 0.138 | [+0.076, +0.203] | sí | 33 / 4 / 23 | ±0.096 |
| RRF k=60, sin filtro | Solo embeddings | nDCG@5 (metadatos) | 60 | 0.132 | [+0.072, +0.191] | sí | 34 / 13 / 13 | ±0.087 |
| RRF k=60 + filtro top-3 (app) | RRF k=60, sin filtro | Hit@5 (documento fuente) | 60 | 0.017 | [-0.067, +0.100] | no | 4 / 3 / 53 | ±0.124 |
| RRF k=60 + filtro top-3 + re-ranker | RRF k=60 + filtro top-3 (app) | Hit@5 (documento fuente) | 60 | 0.017 | [-0.100, +0.133] | no | 7 / 6 / 47 | ±0.170 |
| RRF k=60 + filtro top-3 (app) | RRF k=60, sin filtro | nDCG@5 (metadatos) | 60 | 0.069 | [+0.040, +0.096] | sí | 27 / 7 / 26 | ±0.040 |
| RRF k=60 + filtro top-3 + re-ranker | RRF k=60 + filtro top-3 (app) | nDCG@5 (metadatos) | 60 | 0.026 | [-0.013, +0.063] | no | 24 / 18 / 18 | ±0.057 |
| Híbrido (app) | k-NN 10 | hit3_tipo | 60 | 0.650 | [+0.533, +0.767] | sí | 39 / 0 / 21 | ±0.174 |
| Híbrido (app) | k-NN 10 | hit3_familia | 60 | 0.300 | [+0.167, +0.433] | sí | 20 / 2 / 38 | ±0.192 |

## Detección de consultas sin respuesta

¿Qué tan bien separa cada señal las consultas sin tipo en el corpus de las que sí tienen? AUC 0.5 = azar. El umbral se elige en desarrollo y se configura en COPILOTO_UMBRAL_SIN_PRECEDENTES.

| señal | AUC | umbral sugerido (exactitud balanceada) | detección sin respuesta | falsa alarma en conocidas | n sin respuesta | n conocidas |
|---|---|---|---|---|---|---|
| score_hibrido | 0.570 | 0.585 | 1.000 | 0.633 | 5 | 60 |
| coseno | 0.487 | 0.847 | 1.000 | 0.750 | 5 | 60 |
| kem_max | 0.800 | 0.038 | 1.000 | 0.400 | 5 | 60 |
