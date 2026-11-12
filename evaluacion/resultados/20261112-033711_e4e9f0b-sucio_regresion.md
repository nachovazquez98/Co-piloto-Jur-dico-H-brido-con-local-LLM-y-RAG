# Evaluación: `regresion`

- Fecha: 2026-11-12T03:37:11
- Commit: `e4e9f0b` (con cambios sin confirmar)
- Huella del corpus: `038dbf986378b548` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · semilla 42
- Parámetros: `{"conceptos_v10": "v10", "conceptos_penal": "v11"}`

## Resumen por variante

Aciertos en el top-3 de categorías (sin LLM; conceptos precalculados). Los temas sin tipo en el corpus no cuentan. No es una evaluación: son consultas de ejemplo, no independientes.

| variante | Hit@3 tipo (v10, 18) | Hit@3 familia (penal, 16) |
|---|---|---|
| v10: Jaccard 0.20 + KEM(conceptos) 0.20 | 14/14 | 14/14 |
| Jaccard 0.40, sin KEM | 14/14 | 14/14 |
| KEM(vocabulario) 0.40, sin Jaccard | 13/14 | 14/14 |
| v10 sin lista de tokens | 14/14 | 14/14 |
| Jaccard 0.40, sin KEM, sin lista | 14/14 | 14/14 |
| KEM(vocabulario) 0.40, sin Jaccard, sin lista | 13/14 | 14/14 |
| Configuración actual (config.py) | 14/14 | 14/14 |

## Detalle por ejemplo

Top-3 de cada ejemplo y variante.

| conjunto | variante | ejemplo | acierto | top-3 |
|---|---|---|---|---|
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Lesiones dolosas (riña) | sí | penal/lesiones-dolosas · penal/lesiones-calificadas · penal/violencia-familiar-y-lesiones-dolosas |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/homicidio-a-titulo-de-culpa-y-otro |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/pago-danos-y-perjuicios · civil/accion-de-cumplimiento-de-contrato_prestacion-de-servicios_danos-y-perjuicios |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Daños y perjuicios (accidente vehicular) | sí | civil/accion-de-responsabilidad-civil-objetiva · civil/pago-danos-y-perjuicios · civil/accion-de-dano-moral |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-rescision-de-contrato · civil/accion-rescision-de-contrato_compraventa_otros |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · civil/diligencia-informacion-ad-perpetuam · civil/acreditar-propiedad-de-vehiculo |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Aumento de pensión alimenticia | sí | civil/accion-de-pago-de-pension-alimenticia · familiar/alimentos · familiar/alimentos_menores_pago |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Modificación de guarda y custodia | sí | familiar/guarda-y-custodia · civil/accion-de-fijacion-de-visitas-y-convivencia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Régimen de convivencias | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/visitas-y-convivencias_menor_otras · familiar/guarda-y-custodia_menor |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | delitos sexuales · violencia familiar · trata de personas |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · armas · violencia familiar |
| penal (16) | v10: Jaccard 0.20 + KEM(conceptos) 0.20 | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
| v10 (18) | Jaccard 0.40, sin KEM | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | Jaccard 0.40, sin KEM | Lesiones dolosas (riña) | sí | penal/lesiones-dolosas · penal/lesiones-calificadas · penal/violencia-familiar-y-lesiones-dolosas |
| v10 (18) | Jaccard 0.40, sin KEM | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/homicidio-a-titulo-de-culpa-y-otro |
| v10 (18) | Jaccard 0.40, sin KEM | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | Jaccard 0.40, sin KEM | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | Jaccard 0.40, sin KEM | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | Jaccard 0.40, sin KEM | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | Jaccard 0.40, sin KEM | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/pago-danos-y-perjuicios · civil/accion-de-cumplimiento-de-contrato_prestacion-de-servicios_danos-y-perjuicios |
| v10 (18) | Jaccard 0.40, sin KEM | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | Jaccard 0.40, sin KEM | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | Jaccard 0.40, sin KEM | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | Jaccard 0.40, sin KEM | Daños y perjuicios (accidente vehicular) | sí | civil/accion-de-responsabilidad-civil-objetiva · civil/pago-danos-y-perjuicios · civil/accion-de-dano-moral |
| v10 (18) | Jaccard 0.40, sin KEM | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-rescision-de-contrato · civil/accion-rescision-de-contrato_compraventa_otros |
| v10 (18) | Jaccard 0.40, sin KEM | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · civil/diligencia-informacion-ad-perpetuam · civil/acreditar-propiedad-de-vehiculo |
| v10 (18) | Jaccard 0.40, sin KEM | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | Jaccard 0.40, sin KEM | Aumento de pensión alimenticia | sí | civil/accion-de-pago-de-pension-alimenticia · familiar/alimentos · familiar/alimentos_menores_pago |
| v10 (18) | Jaccard 0.40, sin KEM | Modificación de guarda y custodia | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/guarda-y-custodia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | Jaccard 0.40, sin KEM | Régimen de convivencias | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/visitas-y-convivencias_menor_otras · familiar/guarda-y-custodia_menor |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | delitos sexuales · violencia familiar · trata de personas |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · armas · violencia familiar |
| penal (16) | Jaccard 0.40, sin KEM | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado-en-grado-de-tentativa · penal/robo-calificado |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Lesiones dolosas (riña) | sí | penal/lesiones-calificadas · penal/lesiones-dolosas · penal/lesiones-a-titulo-de-culpa |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/homicidio-a-titulo-de-culpa-y-otro |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-de-responsabilidad-civil-subjetiva · civil/accion-de-responsabilidad-civil-objetiva |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Daños y perjuicios (accidente vehicular) | no | civil/accion-de-responsabilidad-civil-objetiva · civil/accion-de-responsabilidad-civil-objetiva_dano-moral · civil/accion-de-responsabilidad-civil-subjetiva |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-de-responsabilidad-civil-subjetiva · civil/accion-rescision-de-contrato |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · mercantil/accion-de-restitucion-de-inmueble · civil/diligencia-informacion-ad-perpetuam |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Aumento de pensión alimenticia | sí | familiar/alimentos_menores_pago · penal/corrupcion-de-menores-agravado-y-abuso-sexual-infantil-agravado · penal/corrupcion-de-menores-y-abuso-sexual-infantil-agravado |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Modificación de guarda y custodia | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/guarda-y-custodia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard | Régimen de convivencias | sí | familiar/visitas-y-convivencias_menor_otras · penal/sustraccion-de-menores · penal/corrupcion-de-menores |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | sustraccion de menores · delitos sexuales · violencia familiar |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · violencia familiar · armas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
| v10 (18) | v10 sin lista de tokens | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | v10 sin lista de tokens | Lesiones dolosas (riña) | sí | penal/lesiones-dolosas · penal/lesiones-calificadas · penal/violencia-familiar-y-lesiones-dolosas |
| v10 (18) | v10 sin lista de tokens | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/homicidio-a-titulo-de-culpa-y-otro |
| v10 (18) | v10 sin lista de tokens | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | v10 sin lista de tokens | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | v10 sin lista de tokens | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | v10 sin lista de tokens | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | v10 sin lista de tokens | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/pago-danos-y-perjuicios · civil/accion-de-cumplimiento-de-contrato_prestacion-de-servicios_danos-y-perjuicios |
| v10 (18) | v10 sin lista de tokens | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | v10 sin lista de tokens | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | v10 sin lista de tokens | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | v10 sin lista de tokens | Daños y perjuicios (accidente vehicular) | sí | civil/accion-de-responsabilidad-civil-objetiva · civil/pago-danos-y-perjuicios · civil/accion-de-dano-moral |
| v10 (18) | v10 sin lista de tokens | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-rescision-de-contrato · civil/accion-rescision-de-contrato_compraventa_otros |
| v10 (18) | v10 sin lista de tokens | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · civil/diligencia-informacion-ad-perpetuam · civil/acreditar-propiedad-de-vehiculo |
| v10 (18) | v10 sin lista de tokens | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | v10 sin lista de tokens | Aumento de pensión alimenticia | sí | civil/accion-de-pago-de-pension-alimenticia · familiar/alimentos · familiar/alimentos_menores_pago |
| v10 (18) | v10 sin lista de tokens | Modificación de guarda y custodia | sí | familiar/guarda-y-custodia · civil/accion-de-fijacion-de-visitas-y-convivencia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | v10 sin lista de tokens | Régimen de convivencias | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/visitas-y-convivencias_menor_otras · familiar/guarda-y-custodia_menor |
| penal (16) | v10 sin lista de tokens | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | v10 sin lista de tokens | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | v10 sin lista de tokens | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | v10 sin lista de tokens | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | v10 sin lista de tokens | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | v10 sin lista de tokens | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | delitos sexuales · violencia familiar · trata de personas |
| penal (16) | v10 sin lista de tokens | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | v10 sin lista de tokens | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | v10 sin lista de tokens | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | v10 sin lista de tokens | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | v10 sin lista de tokens | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | v10 sin lista de tokens | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | v10 sin lista de tokens | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | v10 sin lista de tokens | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | v10 sin lista de tokens | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · armas · violencia familiar |
| penal (16) | v10 sin lista de tokens | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Lesiones dolosas (riña) | sí | penal/lesiones-dolosas · penal/lesiones-calificadas · penal/violencia-familiar-y-lesiones-dolosas |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/homicidio-a-titulo-de-culpa-y-otro |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/pago-danos-y-perjuicios · civil/accion-de-cumplimiento-de-contrato_prestacion-de-servicios_danos-y-perjuicios |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Daños y perjuicios (accidente vehicular) | sí | civil/accion-de-responsabilidad-civil-objetiva · civil/pago-danos-y-perjuicios · civil/accion-de-dano-moral |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-rescision-de-contrato · civil/accion-rescision-de-contrato_compraventa_otros |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · civil/diligencia-informacion-ad-perpetuam · civil/acreditar-propiedad-de-vehiculo |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Aumento de pensión alimenticia | sí | civil/accion-de-pago-de-pension-alimenticia · familiar/alimentos · familiar/alimentos_menores_pago |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Modificación de guarda y custodia | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/guarda-y-custodia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | Jaccard 0.40, sin KEM, sin lista | Régimen de convivencias | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/visitas-y-convivencias_menor_otras · familiar/guarda-y-custodia_menor |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | delitos sexuales · violencia familiar · trata de personas |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · armas · violencia familiar |
| penal (16) | Jaccard 0.40, sin KEM, sin lista | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado-en-grado-de-tentativa · penal/robo-calificado |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Lesiones dolosas (riña) | sí | penal/lesiones-calificadas · penal/lesiones-dolosas · penal/lesiones-a-titulo-de-culpa |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/responsabilidad-medica-y-homicidio-culposo |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-de-responsabilidad-civil-subjetiva · civil/accion-de-responsabilidad-civil-objetiva |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Daños y perjuicios (accidente vehicular) | no | civil/accion-de-responsabilidad-civil-objetiva · civil/accion-de-responsabilidad-civil-objetiva_dano-moral · civil/accion-de-responsabilidad-civil-subjetiva |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-de-responsabilidad-civil-subjetiva · civil/accion-rescision-de-contrato |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · mercantil/accion-de-restitucion-de-inmueble · civil/diligencia-informacion-ad-perpetuam |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Aumento de pensión alimenticia | sí | familiar/alimentos_menores_pago · penal/corrupcion-de-menores-agravado-y-abuso-sexual-infantil-agravado · penal/corrupcion-de-menores-y-abuso-sexual-infantil-agravado |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Modificación de guarda y custodia | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/guarda-y-custodia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Régimen de convivencias | sí | familiar/visitas-y-convivencias_menor_otras · penal/sustraccion-de-menores · penal/corrupcion-de-menores |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | sustraccion de menores · delitos sexuales · violencia familiar |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · violencia familiar · armas |
| penal (16) | KEM(vocabulario) 0.40, sin Jaccard, sin lista | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
| v10 (18) | Configuración actual (config.py) | Robo calificado | sí | penal/robo-calificado-y-otro · penal/robo-calificado · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | Configuración actual (config.py) | Lesiones dolosas (riña) | sí | penal/lesiones-dolosas · penal/lesiones-calificadas · penal/violencia-familiar-y-lesiones-dolosas |
| v10 (18) | Configuración actual (config.py) | Homicidio culposo | sí | penal/homicidio-culposo · penal/homicidio-a-titulo-de-culpa · penal/homicidio-a-titulo-de-culpa-y-otro |
| v10 (18) | Configuración actual (config.py) | Fraude | sí | penal/fraude-especifico · penal/fraude · penal/fraude-generico |
| v10 (18) | Configuración actual (config.py) | Justicia para adolescentes (robo) | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-calificado-en-grado-de-tentativa |
| v10 (18) | Configuración actual (config.py) | Orden de aprehensión (amparo) | — | penal/fraude-generico · penal/robo-equiparado · penal/abuso-sexual-infantil-agravado-y-otro |
| v10 (18) | Configuración actual (config.py) | Acción cambiaria directa (pagaré) | sí | mercantil/accion-cambiaria-directa · mercantil/accion-cambiaria-directa_cobro-titulo-de-credito · mercantil/accion-de-pago |
| v10 (18) | Configuración actual (config.py) | Incumplimiento de contrato de suministro | — | civil/accion-rescision-de-contrato-de-compraventa · civil/pago-danos-y-perjuicios · civil/accion-de-cumplimiento-de-contrato_prestacion-de-servicios_danos-y-perjuicios |
| v10 (18) | Configuración actual (config.py) | Pago de cheque devuelto | — | penal/robo-calificado · penal/robo-equiparado-en-su-modalidad-de-utilizacion · penal/robo-equiparado |
| v10 (18) | Configuración actual (config.py) | Rescisión de contrato de arrendamiento | sí | civil/accion-rescision-de-contrato-de-arrendamiento · civil/accion-de-terminacion-de-contrato-de-arrendamiento · civil/accion-rescision-de-contrato_arrendamiento |
| v10 (18) | Configuración actual (config.py) | Acción reivindicatoria | sí | civil/accion-reivindicatoria · civil/accion-de-usucapion · civil/diligencias-de-informacion-ad-perpetuam |
| v10 (18) | Configuración actual (config.py) | Daños y perjuicios (accidente vehicular) | sí | civil/accion-de-responsabilidad-civil-objetiva · civil/pago-danos-y-perjuicios · civil/accion-de-dano-moral |
| v10 (18) | Configuración actual (config.py) | Rescisión de compraventa (vicios ocultos) | sí | civil/accion-rescision-de-contrato-de-compraventa · civil/accion-rescision-de-contrato · civil/accion-rescision-de-contrato_compraventa_otros |
| v10 (18) | Configuración actual (config.py) | Información ad perpetuam / usucapión | sí | civil/diligencias-de-informacion-ad-perpetuam · civil/diligencia-informacion-ad-perpetuam · civil/acreditar-propiedad-de-vehiculo |
| v10 (18) | Configuración actual (config.py) | Divorcio | sí | familiar/divorcio_contencioso · civil/accion-de-divorcio · familiar/divorcio_contencioso_otras |
| v10 (18) | Configuración actual (config.py) | Aumento de pensión alimenticia | sí | civil/accion-de-pago-de-pension-alimenticia · familiar/alimentos · familiar/alimentos_menores_pago |
| v10 (18) | Configuración actual (config.py) | Modificación de guarda y custodia | sí | familiar/guarda-y-custodia · civil/accion-de-fijacion-de-visitas-y-convivencia · civil/accion-de-custodia-y-alimentos |
| v10 (18) | Configuración actual (config.py) | Régimen de convivencias | sí | civil/accion-de-fijacion-de-visitas-y-convivencia · familiar/visitas-y-convivencias_menor_otras · familiar/guarda-y-custodia_menor |
| penal (16) | Configuración actual (config.py) | Penal · Robo calificado · Vinculación a proceso | sí | robo · feminicidio · servidores publicos y administracion de justicia |
| penal (16) | Configuración actual (config.py) | Penal · Robo equiparado · Sentencia de juicio oral | sí | robo · fraude y abuso de confianza · armas |
| penal (16) | Configuración actual (config.py) | Penal · Homicidio calificado · Prisión preventiva | sí | homicidio · robo · fraude y abuso de confianza |
| penal (16) | Configuración actual (config.py) | Penal · Homicidio calificado en grado de tentativa · Sentencia de juicio oral | sí | homicidio · feminicidio · lesiones |
| penal (16) | Configuración actual (config.py) | Penal · Violencia familiar · Sentencia de juicio oral | sí | violencia familiar · abandono de familiares · contra la dignidad de las personas |
| penal (16) | Configuración actual (config.py) | Penal · Abuso sexual infantil · Sentencia de juicio oral | sí | delitos sexuales · violencia familiar · trata de personas |
| penal (16) | Configuración actual (config.py) | Penal · Violación · Vinculación a proceso | sí | delitos sexuales · violencia familiar · feminicidio |
| penal (16) | Configuración actual (config.py) | Penal · Fraude · No vinculación (apela el MP) | sí | fraude y abuso de confianza · despojo · abandono de familiares |
| penal (16) | Configuración actual (config.py) | Penal · Despojo de inmuebles · Vinculación a proceso | sí | despojo · abandono de familiares · dano en las cosas |
| penal (16) | Configuración actual (config.py) | Penal · Homicidio culposo (tránsito) · Sentencia de juicio oral | sí | delitos culposos · homicidio · otros |
| penal (16) | Configuración actual (config.py) | Penal · Narcomenudeo · Vinculación a proceso | sí | robo · contra la salud · armas |
| penal (16) | Configuración actual (config.py) | Penal · Abandono de familiares · Sentencia de juicio oral | sí | abandono de familiares · violencia familiar · contra la dignidad de las personas |
| penal (16) | Configuración actual (config.py) | Penal · Secuestro agravado · Sentencia de juicio oral | sí | privacion de la libertad y secuestro · feminicidio · delitos sexuales |
| penal (16) | Configuración actual (config.py) | Penal · Lesiones calificadas · Sentencia de juicio oral | sí | lesiones · violencia familiar · delitos culposos |
| penal (16) | Configuración actual (config.py) | Penal · Justicia para adolescentes · Medida de internamiento (sin tipo en el corpus) | — | robo · armas · violencia familiar |
| penal (16) | Configuración actual (config.py) | Penal · Ejecución de sentencia · Libertad condicionada (sin tipo en el corpus) | — | robo · privacion de la libertad y secuestro · homicidio |
