"""Módulo de evaluación del Co-piloto Jurídico (v11).

- `metricas`: fórmulas (Precision@k, nDCG, MRR, F1 macro, kappa, IC de Wilson y bootstrap).
- `familias`: agrupación de los tipos penales en familias de delito.
- `conjuntos`: consultas, juicios de relevancia, fichas anotadas, plantilla estratificada y detección de fuga.
- `experimentos`: recuperación, ablación, pesos, QC, generación y latencia.
- `reporte`: guardado reproducible de cada corrida.

Uso: `python -m copiloto_juridico.evaluacion --help`
"""
