# Evaluación: `fichas-diagnostico`

- Fecha: 2026-11-20T21:00:27
- Commit: `b21efe0`
- Huella del corpus: `e3c3ff004e3af705` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `sin congelar`
- Parámetros: `{"version_ficha": "v11-9e84ac8f47"}`

> ⚠️ Sin congelado: corrida de desarrollo no atada a una versión.

## Resumen

Diagnóstico sin anotación: un desacuerdo o un artículo ausente no es necesariamente un error del LLM (las reglas también fallan y hay sentencias con páginas omitidas).

| indicador | valor |
|---|---|
| Fichas de esta versión | 2337 |
| JSON inválido tras el reintento | 14/2337 (0.6%) |
| Necesitaron reintento (y quedaron válidas) | 163/2323 (7.0%) |
| Sentido leído por reglas (cobertura) | 1784/2323 (76.8%) |
| Sentido LLM = reglas (donde hay ambos) | 1274/1784 (71.4%) |
| Fichas con algún artículo no encontrado en la sentencia | 36/2323 (1.5%) |
| Artículos citados no encontrados en la sentencia | 44/9622 (0.5%) |
| Segundos por ficha (p50 / p95) | 22 / 37 |

## Sentido del fallo: LLM (filas) vs. resolutivos por reglas (columnas)

La diagonal son coincidencias.

| LLM \ reglas | CONFIRMA | MODIFICA | REPONE | REVOCA |
|---|---|---|---|---|
| CONFIRMA | 859 | 95 | 92 | 218 |
| MODIFICA | 5 | 204 | 3 | 1 |
| REPONE | 0 | 0 | 53 | 0 |
| REVOCA | 1 | 3 | 92 | 158 |

## Dato no disponible por campo

Listas: vacías. Un campo muy vacío es el primero a revisar al anotar (o el texto lo testó).

| campo | dato no disponible |
|---|---|
| tribunal | 0/2323 (0.0%) |
| fecha_resolucion | 426/2323 (18.3%) |
| identificador | 6/2323 (0.3%) |
| etapa_procesal | 10/2323 (0.4%) |
| acto_impugnado | 23/2323 (1.0%) |
| apelante | 97/2323 (4.2%) |
| delitos | 0/2323 (0.0%) |
| hechos_clave | 2/2323 (0.1%) |
| decision | 12/2323 (0.5%) |
| resumen | 0/2323 (0.0%) |
| agravios | 745/2323 (32.1%) |
| fundamento_legal | 66/2323 (2.8%) |
| pruebas_decisivas | 564/2323 (24.3%) |
| sentido_fallo = OTRO | 0/2323 (0.0%) |
