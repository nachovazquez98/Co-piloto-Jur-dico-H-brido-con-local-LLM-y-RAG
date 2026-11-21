# Evaluación: `familias-coherencia`

- Fecha: 2026-11-21T03:26:30
- Commit: `4b1529e`
- Huella del corpus: `77466c884916dd0e` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `a15b83c852281010`
- Parámetros: `{"min_sentencias_por_tipo": 3}`

## Resumen

Tipos con al menos 3 sentencias cuyo centroide de embeddings queda más cerca de su propia familia (calculada sin ese tipo) que de cualquier otra. Es una comprobación de consistencia sin LLM; no dice qué agrupación es la jurídicamente correcta.

| tipos | más cerca de su familia | más cerca de otra |
|---|---|---|
| 246 | 166 | 80 |

## Tipos más cercanos a otra familia

Posibles errores de agrupación o tipos con varios delitos.

| tipo | sentencias | familia asignada | familia más cercana | similitud asignada | similitud más cercana |
|---|---|---|---|---|---|
| penal/amenazas | 23 | amenazas-y-extorsion | contra-la-dignidad-de-las-personas | 0.705 | 0.897 |
| penal/parricidio | 23 | homicidio | feminicidio | 0.418 | 0.477 |
| penal/contra-la-dignidad-de-las-personas | 20 | contra-la-dignidad-de-las-personas | amenazas-y-extorsion | 0.683 | 0.855 |
| penal/maltrato-infantil | 16 | violencia-familiar | sustraccion-de-menores | 0.720 | 0.750 |
| penal/robo-calificado-y-otro | 15 | robo | servidores-publicos-y-administracion-de-justicia | 0.706 | 0.727 |
| penal/extorsion | 14 | amenazas-y-extorsion | servidores-publicos-y-administracion-de-justicia | 0.633 | 0.705 |
| penal/privacion-ilegal-de-la-libertad | 11 | privacion-de-la-libertad-y-secuestro | amenazas-y-extorsion | 0.122 | 0.677 |
| penal/abigeato-calificado | 10 | robo | armas | 0.508 | 0.588 |
| penal/falsedad-en-declaraciones-y-en-informes-dados-a-una-autoridad | 10 | falsedad-y-falsificacion | fraude-y-abuso-de-confianza | 0.739 | 0.796 |
| penal/tentantiva-de-robo-calificado | 10 | robo | armas | 0.569 | 0.611 |
| penal/dano-en-las-cosas-a-titulo-de-culpa | 10 | delitos-culposos | dano-en-las-cosas | 0.746 | 0.775 |
| penal/falsificacion-y-uso-indebido-de-sellos-marcas-llaves-y-troqueles | 9 | falsedad-y-falsificacion | robo | 0.390 | 0.703 |
| penal/adquisicion-ilegitima-de-bienes-materia-de-un-delito-o-una-fraccion-penal | 9 | otros | robo | 0.197 | 0.403 |
| penal/aborto | 9 | otros | responsabilidad-profesional | -0.048 | 0.278 |
| penal/violencia-familiar-y-lesiones-dolosas | 9 | violencia-familiar | contra-la-dignidad-de-las-personas | 0.814 | 0.845 |
| penal/violencia-intrafamiliar-y-lesiones | 9 | violencia-familiar | lesiones | 0.739 | 0.768 |
| penal/feminicidio-en-grado-de-tentativa | 9 | feminicidio | contra-la-dignidad-de-las-personas | 0.766 | 0.766 |
| penal/extorsion-y-otros | 9 | amenazas-y-extorsion | cuestiones-procesales | 0.539 | 0.615 |
| penal/robo-calificado-y-lesiones | 9 | robo | lesiones | 0.516 | 0.572 |
| penal/homicidio-calificado-y-robo-calificado | 8 | homicidio | amenazas-y-extorsion | 0.293 | 0.579 |
| penal/secuestro-y-delincuencia-organizada | 8 | privacion-de-la-libertad-y-secuestro | servidores-publicos-y-administracion-de-justicia | 0.320 | 0.589 |
| penal/lesiones-culposas | 7 | delitos-culposos | responsabilidad-profesional | 0.373 | 0.613 |
| penal/tentativa-de-robo-calificado | 7 | robo | armas | 0.559 | 0.654 |
| penal/homicidio | 7 | homicidio | amenazas-y-extorsion | 0.261 | 0.685 |
| penal/delitos-cometidos-contra-representantes-de-la-autoridad | 6 | otros | delincuencia-organizada | -0.362 | 0.404 |
| penal/danos-en-las-cosas-a-titulo-de-culpa | 6 | delitos-culposos | dano-en-las-cosas | 0.592 | 0.802 |
| penal/hostigamiento-sexual | 6 | delitos-sexuales | contra-la-dignidad-de-las-personas | 0.711 | 0.791 |
| penal/secuestro-y-asociacion-delictuosa | 6 | privacion-de-la-libertad-y-secuestro | cuestiones-procesales | 0.427 | 0.525 |
| penal/usurpacion-de-funciones-publicas-o-de-profesion-y-uso-indebido-de-uniformes-o-insignias | 5 | otros | falsedad-y-falsificacion | 0.517 | 0.667 |
| penal/dano-en-las-cosas-y-lesiones | 5 | dano-en-las-cosas | delitos-culposos | 0.660 | 0.836 |
| penal/asociacion-delictuosa-y-otros | 5 | delincuencia-organizada | falsedad-y-falsificacion | 0.184 | 0.436 |
| penal/allanamiento-de-morada | 5 | otros | despojo | 0.369 | 0.607 |
| penal/responsabilidad-profesional-y-tecnica | 5 | responsabilidad-profesional | violencia-familiar | 0.489 | 0.660 |
| penal/desvio-y-aprovechamiento-indebido-de-atribuciones-y-facultades | 5 | servidores-publicos-y-administracion-de-justicia | fraude-y-abuso-de-confianza | 0.661 | 0.755 |
| penal/homicidio-en-rina-inesperada | 5 | homicidio | delincuencia-organizada | 0.117 | 0.260 |
| penal/homicidio-calificado-y-dano-en-las-cosas-a-titulo-de-culpa | 5 | delitos-culposos | delincuencia-organizada | -0.119 | 0.282 |
| penal/violencia-intrafamiliar-equiparada | 5 | violencia-familiar | otros | 0.499 | 0.609 |
| penal/violacion-y-violencia-intrafamiliar | 5 | delitos-sexuales | violencia-familiar | 0.683 | 0.764 |
| penal/robo | 5 | robo | fraude-y-abuso-de-confianza | 0.585 | 0.711 |
| penal/dano-en-las-cosas-y-otros | 4 | dano-en-las-cosas | despojo | 0.421 | 0.496 |
| penal/violencia-familiar-y-abandono-de-familiares | 4 | violencia-familiar | abandono-de-familiares | 0.663 | 0.767 |
| penal/lesiones-y-violencia-intrafamiliar | 4 | lesiones | violencia-familiar | 0.626 | 0.774 |
| penal/lesiones-y-danos-en-las-cosas | 4 | lesiones | delitos-culposos | 0.518 | 0.661 |
| penal/contra-la-salud-y-otro | 4 | contra-la-salud | armas | 0.464 | 0.594 |
| penal/falsificacion-de-documentos-en-general | 4 | falsedad-y-falsificacion | fraude-y-abuso-de-confianza | 0.683 | 0.729 |
| penal/lesiones-a-titulo-de-culpa-grave | 4 | delitos-culposos | lesiones | 0.688 | 0.721 |
| penal/disparo-de-arma-de-fuego-sobre-persona-y-delitos-cometidos-contra-representantes-de-la-autoridad | 4 | armas | delincuencia-organizada | 0.040 | 0.187 |
| penal/desobediencia-o-resistencia-de-particulares-y-otro | 4 | otros | amenazas-y-extorsion | 0.355 | 0.689 |
| penal/homicidio-en-grado-de-tentativa | 4 | homicidio | lesiones | 0.277 | 0.572 |
| penal/suplantacion-de-identidad | 4 | falsedad-y-falsificacion | fraude-y-abuso-de-confianza | 0.578 | 0.614 |
| penal/abuso-de-confianza-equiparado | 3 | fraude-y-abuso-de-confianza | amenazas-y-extorsion | 0.510 | 0.637 |
| penal/dano-a-las-cosas | 3 | dano-en-las-cosas | cuestiones-procesales | 0.514 | 0.554 |
| penal/delitos-cometidos-en-la-administracion-de-justicia-y-en-otros-ramos-del-poder-publico-a-titulo-de-dolo | 3 | servidores-publicos-y-administracion-de-justicia | cuestiones-procesales | 0.615 | 0.621 |
| penal/homicidio-calificado-y-delincuencia-organizada | 3 | homicidio | delincuencia-organizada | 0.284 | 0.289 |
| penal/homicidio-calificado-y-parricidio | 3 | homicidio | armas | 0.351 | 0.403 |
| penal/fraude-y-otros | 3 | fraude-y-abuso-de-confianza | contra-la-dignidad-de-las-personas | 0.619 | 0.643 |
| penal/desobediencia-o-resistencia-de-particulares | 3 | otros | servidores-publicos-y-administracion-de-justicia | 0.445 | 0.665 |
| penal/evasion-de-presos | 3 | otros | amenazas-y-extorsion | 0.492 | 0.656 |
| penal/disparo-de-arma-de-fuego-y-otros | 3 | armas | amenazas-y-extorsion | 0.400 | 0.493 |
| penal/disparo-de-arma-de-fuego-sobre-persona-y-otros | 3 | armas | lesiones | 0.433 | 0.558 |
| penal/falsificacion-de-documentos-en-general-y-otro | 3 | falsedad-y-falsificacion | fraude-y-abuso-de-confianza | 0.750 | 0.761 |
| penal/encubrimiento | 3 | servidores-publicos-y-administracion-de-justicia | feminicidio | 0.242 | 0.326 |
| penal/fraude-especifico-y-otro | 3 | fraude-y-abuso-de-confianza | despojo | 0.304 | 0.478 |
| penal/homicidio-en-grado-de-tentativa-y-otros | 3 | homicidio | armas | 0.239 | 0.323 |
| penal/lesiones-dolosas-y-violencia-intrafamiliar | 3 | lesiones | violencia-familiar | 0.087 | 0.125 |
| penal/instigacion-o-ayuda-al-suicidio | 3 | homicidio | feminicidio | 0.003 | 0.427 |
| penal/homicidio-en-rina-y-otros | 3 | homicidio | lesiones | 0.428 | 0.605 |
| penal/homicidio-simple-intencional-y-robo-calificado | 3 | homicidio | armas | 0.239 | 0.395 |
| penal/lesiones-y-dano-en-las-cosas | 3 | lesiones | delitos-culposos | 0.378 | 0.678 |
| penal/secuestro-express-agravado | 3 | privacion-de-la-libertad-y-secuestro | delincuencia-organizada | 0.248 | 0.289 |
| penal/robo-calificado-y-corrupcion-de-menores | 3 | robo | delincuencia-organizada | -0.053 | 0.461 |
| penal/robo-equiparado-en-su-modalidad-de-utilizar-los-automotores-robados-a-sabiendas-de-su-origen-ilicito | 3 | robo | armas | 0.361 | 0.479 |
| penal/robo-equiparado-y-otro | 3 | robo | contra-la-salud | 0.511 | 0.541 |
| penal/uso-ilicito-de-atribuciones-y-facultades | 3 | servidores-publicos-y-administracion-de-justicia | amenazas-y-extorsion | 0.705 | 0.758 |
| penal/tentativa-de-homicidio-calificado-y-robo-calificado | 3 | homicidio | robo | 0.294 | 0.500 |
| penal/tentativa-de-homicidio-calificado-y-otro | 3 | homicidio | servidores-publicos-y-administracion-de-justicia | 0.272 | 0.517 |
| penal/secuestro-y-extorsion | 3 | privacion-de-la-libertad-y-secuestro | fraude-y-abuso-de-confianza | 0.377 | 0.430 |
| penal/ultrajes-a-la-moral-y-a-las-buenas-costumbres | 3 | otros | contra-la-dignidad-de-las-personas | 0.418 | 0.660 |
| penal/violencia-intrafamiliar-y-abandono-de-familiares | 3 | violencia-familiar | abandono-de-familiares | 0.510 | 0.736 |
| penal/violencia-intrafamiliar-y-lesiones-simples | 3 | violencia-familiar | lesiones | 0.573 | 0.680 |
