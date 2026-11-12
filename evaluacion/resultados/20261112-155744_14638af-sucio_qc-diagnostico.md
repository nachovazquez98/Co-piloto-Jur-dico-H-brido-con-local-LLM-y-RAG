# Evaluación: `qc-diagnostico`

- Fecha: 2026-11-12T15:57:44
- Commit: `14638af` (con cambios sin confirmar)
- Huella del corpus: `038dbf986378b548` · embeddings `intfloat/multilingual-e5-base`
- LLM: `llama3.1:8b` · semilla 42
- Parámetros: `{"k": 5, "consultas": "16 ejemplos penales de la app"}`

## Tasa de descarte de QC_1 por versión y contexto

Para los 5 primeros candidatos de cada uno de los 16 ejemplos penales. Solo diagnóstico: un descarte puede ser correcto o injusto; la calidad se mide con `qc` contra juicios humanos.

| prompt QC_1 | contexto | pares | descartados (NO) | ambiguos | consultas con los 5 descartados |
|---|---|---|---|---|---|
| v10 | inicio | 80 | 0.537 | 0.000 | 4 |
| v10 | fragmento_similar | 80 | 0.575 | 0.000 | 1 |
| v11 | inicio | 80 | 0.212 | 0.000 | 0 |
| v11 | fragmento_similar | 80 | 0.150 | 0.000 | 0 |
