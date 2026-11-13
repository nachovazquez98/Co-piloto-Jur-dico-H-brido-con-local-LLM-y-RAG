# Evaluación: `fichas-diagnostico`

- Fecha: 2026-11-13T10:50:03
- Commit: `fb1f3fe`
- Huella del corpus: `bea10929ce634a58` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `sin congelar`
- Parámetros: `{"version_ficha": "v11-9e84ac8f47"}`

> ⚠️ Sin congelado: corrida de desarrollo no atada a una versión.

## Resumen

Diagnóstico sin anotación: un desacuerdo o un artículo ausente no es necesariamente un error del LLM (las reglas también fallan y hay sentencias con páginas omitidas).

| indicador | valor |
|---|---|
| Fichas de esta versión | 1412 |
| JSON inválido tras el reintento | 66/1412 (4.7%) |
| Necesitaron reintento (y quedaron válidas) | 28/1346 (2.1%) |
| Sentido leído por reglas (cobertura) | 538/1346 (40.0%) |
| Sentido LLM = reglas (donde hay ambos) | 349/538 (64.9%) |
| Fichas con algún artículo no encontrado en la sentencia | 259/1346 (19.2%) |
| Artículos citados no encontrados en la sentencia | 429/5096 (8.4%) |
| Segundos por ficha (p50 / p95) | 26 / 40 |

## Sentido del fallo: LLM (filas) vs. resolutivos por reglas (columnas)

La diagonal son coincidencias.

| LLM \ reglas | CONFIRMA | MODIFICA | REPONE | REVOCA |
|---|---|---|---|---|
| CONFIRMA | 241 | 9 | 44 | 93 |
| MODIFICA | 2 | 40 | 2 | 0 |
| REPONE | 0 | 0 | 19 | 0 |
| REVOCA | 0 | 1 | 38 | 49 |

## Dato no disponible por campo

Listas: vacías. Un campo muy vacío es el primero a revisar al anotar (o el texto lo testó).

| campo | dato no disponible |
|---|---|
| tribunal | 0/1346 (0.0%) |
| fecha_resolucion | 332/1346 (24.7%) |
| identificador | 3/1346 (0.2%) |
| etapa_procesal | 9/1346 (0.7%) |
| acto_impugnado | 19/1346 (1.4%) |
| apelante | 52/1346 (3.9%) |
| delitos | 0/1346 (0.0%) |
| hechos_clave | 0/1346 (0.0%) |
| decision | 10/1346 (0.7%) |
| resumen | 0/1346 (0.0%) |
| agravios | 457/1346 (34.0%) |
| fundamento_legal | 37/1346 (2.7%) |
| pruebas_decisivas | 307/1346 (22.8%) |
| sentido_fallo = OTRO | 0/1346 (0.0%) |
