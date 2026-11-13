# Evaluación: `qc-diagnostico`

- Fecha: 2026-11-13T11:44:12
- Commit: `fb1f3fe`
- Huella del corpus: `bea10929ce634a58` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `sin congelar`
- Parámetros: `{"k": 5, "consultas": "16 ejemplos penales de la app"}`

> ⚠️ Sin congelado: corrida de desarrollo no atada a una versión.

## Tasa de descarte de QC_1 por versión y contexto

Para los 5 primeros candidatos de cada uno de los 16 ejemplos penales. Solo diagnóstico: un descarte puede ser correcto o injusto; la calidad se mide con `qc` contra juicios humanos.

| prompt QC_1 | contexto | pares | descartados (NO) | ambiguos | consultas con los 5 descartados |
|---|---|---|---|---|---|
| v10 | inicio | 80 | 0.512 | 0.000 | 2 |
| v10 | fragmento_similar | 80 | 0.550 | 0.000 | 1 |
| v11 | inicio | 80 | 0.163 | 0.000 | 1 |
| v11 | fragmento_similar | 80 | 0.138 | 0.000 | 0 |

## Veredictos de QC_1 por candidato (configuración de la app)

Prompt v11 · contexto fragmento_similar. Lo usa el comando `errores`.

| ejemplo | posición | doc_id | tipo | veredicto |
|---|---|---|---|---|
| Penal · Robo calificado · Vinculación a proceso | 1 | 2025-05-13-100223_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Robo calificado · Vinculación a proceso | 2 | 2025-04-24-105911_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Robo calificado · Vinculación a proceso | 3 | 2025-08-14-131843_penal_robo-calificado-en-grado-de-tentativa.pdf | penal/robo-calificado-en-grado-de-tentativa | SI |
| Penal · Robo calificado · Vinculación a proceso | 4 | 2025-04-23-132527_penal_robo-calificado-en-grado-de-tentativa.pdf | penal/robo-calificado-en-grado-de-tentativa | SI |
| Penal · Robo calificado · Vinculación a proceso | 5 | 2025-11-05-140542_penal_robo-calificado-y-otro.pdf | penal/robo-calificado-y-otro | SI |
| Penal · Robo equiparado · Sentencia de juicio oral | 1 | 2025-04-10-130540_penal_robo-equiparado-en-su-modalidad-de-utilizacion.pdf | penal/robo-equiparado-en-su-modalidad-de-utilizacion | SI |
| Penal · Robo equiparado · Sentencia de juicio oral | 2 | 2025-11-04-143146_penal_robo-calificado-y-portacion-de-armas-y-objetos-prohibidos.pdf | penal/robo-calificado-y-portacion-de-armas-y-objetos-prohibidos | SI |
| Penal · Robo equiparado · Sentencia de juicio oral | 3 | 2025-05-13-100223_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Robo equiparado · Sentencia de juicio oral | 4 | 2025-07-01-095837_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Robo equiparado · Sentencia de juicio oral | 5 | 2025-08-25-132412_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Homicidio calificado · Prisión preventiva | 1 | 2025-06-10-111734_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | SI |
| Penal · Homicidio calificado · Prisión preventiva | 2 | 2025-10-06-113638_penal_instigacion-o-ayuda-al-suicidio.pdf | penal/instigacion-o-ayuda-al-suicidio | SI |
| Penal · Homicidio calificado · Prisión preventiva | 3 | 2025-09-09-135130_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | SI |
| Penal · Homicidio calificado · Prisión preventiva | 4 | 2025-11-27-144957_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | SI |
| Penal · Homicidio calificado · Prisión preventiva | 5 | 2025-08-22-123252_penal_homicidio-simple-intencional-en-grado-de-tentativa.pdf | penal/homicidio-simple-intencional-en-grado-de-tentativa | SI |
| Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | 1 | 2025-11-10-103321_penal_lesiones-calificadas.pdf | penal/lesiones-calificadas | SI |
| Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | 2 | 2025-07-07-184748_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | SI |
| Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | 3 | 2025-08-06-102352_penal_tentativa-de-homicidio-calificado.pdf | penal/tentativa-de-homicidio-calificado | NO |
| Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | 4 | 2025-11-27-135349_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | SI |
| Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | 5 | 2025-04-18-112111_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | SI |
| Penal · Violencia familiar · Sentencia de juicio oral | 1 | 2025-04-08-154222_penal_violencia-intrafamiliar.pdf | penal/violencia-intrafamiliar | SI |
| Penal · Violencia familiar · Sentencia de juicio oral | 2 | 2025-03-31-113456_penal_maltrato-infantil.pdf | penal/maltrato-infantil | SI |
| Penal · Violencia familiar · Sentencia de juicio oral | 3 | 2025-12-11-145211_penal_violencia-familiar.pdf | penal/violencia-familiar | SI |
| Penal · Violencia familiar · Sentencia de juicio oral | 4 | 2025-12-12-142128_penal_violencia-familiar-y-amenazas.pdf | penal/violencia-familiar-y-amenazas | SI |
| Penal · Violencia familiar · Sentencia de juicio oral | 5 | 2025-09-18-140956_penal_violencia-familiar-y-otros.pdf | penal/violencia-familiar-y-otros | SI |
| Penal · Abuso sexual infantil · Sentencia de juicio oral | 1 | 2025-10-07-091955_penal_abuso-sexual-infantil-agravado.pdf | penal/abuso-sexual-infantil-agravado | SI |
| Penal · Abuso sexual infantil · Sentencia de juicio oral | 2 | 2025-05-12-113442_penal_abuso-sexual-infantil-agravado.pdf | penal/abuso-sexual-infantil-agravado | SI |
| Penal · Abuso sexual infantil · Sentencia de juicio oral | 3 | 2025-06-13-130122_penal_abuso-sexual-infantil.pdf | penal/abuso-sexual-infantil | SI |
| Penal · Abuso sexual infantil · Sentencia de juicio oral | 4 | 2025-03-21-142930_penal_abuso-sexual-infantil-agravado.pdf | penal/abuso-sexual-infantil-agravado | NO |
| Penal · Abuso sexual infantil · Sentencia de juicio oral | 5 | 2025-10-01-125656_penal_abuso-sexual-infantil-agravado-y-corrupcion-de-menores.pdf | penal/abuso-sexual-infantil-agravado-y-corrupcion-de-menores | SI |
| Penal · Violación · Vinculación a proceso | 1 | 2025-08-07-164051_penal_abuso-sexual-infantil-agravado.pdf | penal/abuso-sexual-infantil-agravado | NO |
| Penal · Violación · Vinculación a proceso | 2 | 2025-08-26-120921_penal_abuso-sexual.pdf | penal/abuso-sexual | SI |
| Penal · Violación · Vinculación a proceso | 3 | 2025-07-11-121010_penal_violacion.pdf | penal/violacion | SI |
| Penal · Violación · Vinculación a proceso | 4 | 2025-03-31-094135_penal_violacion.pdf | penal/violacion | SI |
| Penal · Violación · Vinculación a proceso | 5 | 2026-01-06-151037_penal_violacion.pdf | penal/violacion | SI |
| Penal · Fraude · No vinculación (apela el MP) | 1 | 2025-08-12-112910_penal_fraude-especifico.pdf | penal/fraude-especifico | SI |
| Penal · Fraude · No vinculación (apela el MP) | 2 | 2025-12-19-142059_penal_fraude-especifico.pdf | penal/fraude-especifico | SI |
| Penal · Fraude · No vinculación (apela el MP) | 3 | 2025-04-03-120507_penal_fraude-generico.pdf | penal/fraude-generico | SI |
| Penal · Fraude · No vinculación (apela el MP) | 4 | 2025-11-26-125922_penal_fraude-procesal.pdf | penal/fraude-procesal | SI |
| Penal · Fraude · No vinculación (apela el MP) | 5 | 2025-04-21-130659_penal_fraude-generico.pdf | penal/fraude-generico | NO |
| Penal · Despojo de inmuebles · Vinculación a proceso | 1 | 2025-10-01-133313_penal_despojo-de-inmueble.pdf | penal/despojo-de-inmueble | SI |
| Penal · Despojo de inmuebles · Vinculación a proceso | 2 | 2025-04-09-094938_penal_despojo-y-dano-en-las-cosas.pdf | penal/despojo-y-dano-en-las-cosas | SI |
| Penal · Despojo de inmuebles · Vinculación a proceso | 3 | 2025-09-11-145653_penal_despojo-de-inmuebles-y-aguas.pdf | penal/despojo-de-inmuebles-y-aguas | SI |
| Penal · Despojo de inmuebles · Vinculación a proceso | 4 | 2025-09-12-105626_penal_despojo-de-inmuebles.pdf | penal/despojo-de-inmuebles | SI |
| Penal · Despojo de inmuebles · Vinculación a proceso | 5 | 2025-05-30-120851_penal_despojo-de-inmueble.pdf | penal/despojo-de-inmueble | SI |
| Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | 1 | 2025-06-10-100359_penal_homicidio-a-titulo-de-culpa.pdf | penal/homicidio-a-titulo-de-culpa | SI |
| Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | 2 | 2025-07-08-143912_penal_homicidio-culposo.pdf | penal/homicidio-culposo | SI |
| Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | 3 | 2025-06-13-100453_penal_homicidio-a-titulo-de-culpa-y-dano-en-las-cosas-a-titulo-de-culpa.pdf | penal/homicidio-a-titulo-de-culpa-y-dano-en-las-cosas-a-titulo-de-culpa | SI |
| Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | 4 | 2025-03-21-144847_penal_homicidio-culposo.pdf | penal/homicidio-culposo | SI |
| Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | 5 | 2025-09-26-113100_penal_homicidio-y-dano-en-las-cosas-a-titulo-de-culpa.pdf | penal/homicidio-y-dano-en-las-cosas-a-titulo-de-culpa | SI |
| Penal · Narcomenudeo · Vinculación a proceso | 1 | 2025-06-04-110914_penal_contra-la-salud-en-su-modalidad-de-narcomenudeo.pdf | penal/contra-la-salud-en-su-modalidad-de-narcomenudeo | SI |
| Penal · Narcomenudeo · Vinculación a proceso | 2 | 2025-06-30-100657_penal_narcomenudeo.pdf | penal/narcomenudeo | SI |
| Penal · Narcomenudeo · Vinculación a proceso | 3 | 2025-09-23-091959_penal_robo-equiparado-en-su-modalidad-de-desmantelamiento.pdf | penal/robo-equiparado-en-su-modalidad-de-desmantelamiento | SI |
| Penal · Narcomenudeo · Vinculación a proceso | 4 | 2025-06-30-101727_penal_robo-equiparado.pdf | penal/robo-equiparado | SI |
| Penal · Narcomenudeo · Vinculación a proceso | 5 | 2025-05-30-133029_penal_narcomenudeo.pdf | penal/narcomenudeo | SI |
| Penal · Abandono de familiares · Sentencia de juicio oral | 1 | 2025-09-10-114715_penal_abandono-de-familiares.pdf | penal/abandono-de-familiares | SI |
| Penal · Abandono de familiares · Sentencia de juicio oral | 2 | 2025-08-28-115114_penal_abandono-de-familiares.pdf | penal/abandono-de-familiares | SI |
| Penal · Abandono de familiares · Sentencia de juicio oral | 3 | 2025-08-12-104417_penal_abandono-de-familiares.pdf | penal/abandono-de-familiares | SI |
| Penal · Abandono de familiares · Sentencia de juicio oral | 4 | 2025-10-29-130912_penal_abandono-de-familiares.pdf | penal/abandono-de-familiares | NO |
| Penal · Abandono de familiares · Sentencia de juicio oral | 5 | 2025-07-15-134339_penal_abandono-de-familiares.pdf | penal/abandono-de-familiares | SI |
| Penal · Secuestro agravado · Sentencia de juicio oral | 1 | 2025-12-12-141255_penal_secuestro-agravado-y-otro.pdf | penal/secuestro-agravado-y-otro | SI |
| Penal · Secuestro agravado · Sentencia de juicio oral | 2 | 2025-06-05-135030_penal_secuestro-express.pdf | penal/secuestro-express | NO |
| Penal · Secuestro agravado · Sentencia de juicio oral | 3 | 2025-08-29-152021_penal_abuso-sexual-infantil-agravado.pdf | penal/abuso-sexual-infantil-agravado | NO |
| Penal · Secuestro agravado · Sentencia de juicio oral | 4 | 2025-10-07-133710_penal_secuestro-agravado.pdf | penal/secuestro-agravado | SI |
| Penal · Secuestro agravado · Sentencia de juicio oral | 5 | 2025-04-01-132512_penal_abuso-sexual-infantil-agravado.pdf | penal/abuso-sexual-infantil-agravado | NO |
| Penal · Lesiones calificadas · Sentencia de juicio oral | 1 | 2025-03-17-124454_penal_homicidio-en-rina.pdf | penal/homicidio-en-rina | SI |
| Penal · Lesiones calificadas · Sentencia de juicio oral | 2 | 2025-04-01-181604_penal_lesiones-calificadas.pdf | penal/lesiones-calificadas | SI |
| Penal · Lesiones calificadas · Sentencia de juicio oral | 3 | 2025-04-08-110423_penal_homicidio-simple.pdf | penal/homicidio-simple | SI |
| Penal · Lesiones calificadas · Sentencia de juicio oral | 4 | 2025-07-07-184748_penal_homicidio-calificado-en-grado-de-tentativa.pdf | penal/homicidio-calificado-en-grado-de-tentativa | NO |
| Penal · Lesiones calificadas · Sentencia de juicio oral | 5 | 2025-06-03-123253_penal_tentativa-de-homicidio-calificado-y-tentativa-de-robo-calificado.pdf | penal/tentativa-de-homicidio-calificado-y-tentativa-de-robo-calificado | SI |
| Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | 1 | 2025-03-31-114853_penal_tentativa-de-robo-calificado.pdf | penal/tentativa-de-robo-calificado | SI |
| Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | 2 | 2025-09-23-132730_penal_robo-calificado-tentado.pdf | penal/robo-calificado-tentado | SI |
| Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | 3 | 2025-09-10-141345_penal_robo-calificado-y-otro.pdf | penal/robo-calificado-y-otro | NO |
| Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | 4 | 2025-04-24-122020_penal_robo-simple.pdf | penal/robo-simple | SI |
| Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | 5 | 2025-04-16-122721_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | 1 | 2025-10-03-212533_penal_lesiones-calificadas.pdf | penal/lesiones-calificadas | SI |
| Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | 2 | 2025-09-23-131006_penal_lesiones-y-amenazas.pdf | penal/lesiones-y-amenazas | NO |
| Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | 3 | 2025-04-16-122721_penal_robo-calificado.pdf | penal/robo-calificado | SI |
| Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | 4 | 2025-10-03-214656_penal_robo-calificado-en-grado-de-tentativa.pdf | penal/robo-calificado-en-grado-de-tentativa | SI |
| Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | 5 | 2025-04-28-114153_penal_robo-calificado.pdf | penal/robo-calificado | SI |
