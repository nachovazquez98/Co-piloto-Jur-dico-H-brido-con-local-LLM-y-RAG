# Evaluación: `metadatos-cobertura`

- Fecha: 2026-11-20T01:44:23
- Commit: `c27877e`
- Huella del corpus: `e3c3ff004e3af705` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` · semilla del LLM 42 · semilla de evaluación 42
- Congelado: `sin congelar`
- Parámetros: `{"sentencias_penales": 2495}`

## Cobertura de los metadatos procesales (reglas)

Valores leídos por reglas del párrafo inicial (acto impugnado, quién apeló) y de los puntos resolutivos (sentido). 'no determinado' = las reglas no lo encontraron. Sin anotación: no mide exactitud.

| campo | valor | sentencias | fracción |
|---|---|---|---|
| acto_impugnado | otro | 661 | 0.265 |
| acto_impugnado | sentencia definitiva | 468 | 0.188 |
| acto_impugnado | no determinado | 462 | 0.185 |
| acto_impugnado | no vinculación | 392 | 0.157 |
| acto_impugnado | vinculación | 341 | 0.137 |
| acto_impugnado | medida cautelar | 107 | 0.043 |
| acto_impugnado | sobreseimiento | 64 | 0.026 |
| apelante | varios | 630 | 0.253 |
| apelante | defensa | 602 | 0.241 |
| apelante | ministerio público | 580 | 0.232 |
| apelante | no determinado | 510 | 0.204 |
| apelante | asesor de la víctima | 173 | 0.069 |
| sentido | CONFIRMA | 968 | 0.388 |
| sentido | no determinado | 529 | 0.212 |
| sentido | REVOCA | 404 | 0.162 |
| sentido | MODIFICA | 332 | 0.133 |
| sentido | REPONE | 262 | 0.105 |
