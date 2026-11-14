<div align="center">

# ⚖️ Co-piloto Jurídico

**Búsqueda y análisis de precedentes penales del Supremo Tribunal de Justicia de Jalisco para la defensa en
apelación, con búsqueda híbrida (embeddings + BM25), un LLM local y una interfaz web.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App_web-ff4b4b?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-LLM_local-blueviolet)
![Version](https://img.shields.io/badge/versión-11.0.0-informational)
![License](https://img.shields.io/badge/licencia-MIT-green)

</div>

---

## Para quién es y qué hace

**Usuario:** abogado de la defensa penal (pública o particular) que prepara o contesta un recurso de apelación ante
las Salas Penales del STJ Jalisco.

**Qué hace:** recibe la descripción del caso en lenguaje natural y devuelve las sentencias penales del propio tribunal
más parecidas, con una ficha de cada una (delito, acto impugnado, agravios y cómo los calificó la Sala, sentido del
fallo, fundamento legal) y una síntesis comparativa.

**Qué no hace:** no predice el resultado de un juicio, no redacta agravios y no sustituye la lectura de la sentencia
original ni el criterio del abogado. Las resoluciones del STJ son precedentes orientadores, no jurisprudencia
obligatoria.

## Uso en 4 pasos

1. Instala y abre la app (ver [Instalación](#instalación)): `streamlit run app/Copiloto.py`.
2. Escribe el caso (hechos, delito, qué resolución se apela y por qué) o elige un ejemplo.
3. Pulsa **Buscar precedentes** y espera a que terminen los 5 pasos.
4. Lee la síntesis (empieza con qué precedentes fueron favorables o desfavorables para la defensa) y las tarjetas.
   Donde aparezca **verificar** (el sentido del fallo de la ficha no coincide con los resolutivos, o un artículo
   citado no está en la sentencia) o **no verificado por QC_1**, revisa la sentencia original antes de citarla.
   Puedes descargar el informe en Markdown.

La página **Cómo funciona** explica el flujo, los límites de tamaño, un glosario jurídico breve y muestra la última
evaluación guardada.

## Cómo funciona (resumen)

| Paso | Entrada → salida | Técnica |
|---|---|---|
| 1. Conceptos clave | consulta → delito + problema procesal | LLM (Ollama, `llama3.1:8b`) |
| 2. Clasificación | consulta → 3 familias de delito más probables | coseno de embeddings + Jaccard de lemas + voto de las sentencias recuperadas |
| 3. Búsqueda | familias → hasta 30 sentencias candidatas | embeddings `multilingual-e5-base` (ChromaDB) + BM25, fusión RRF; filtro por familia con respaldo sin filtro |
| 4. Fichas | candidato → ficha penal | QC_1 (LLM: ¿sirve como precedente?), ficha en **una** llamada (JSON validado), verificaciones deterministas (sentido del fallo por reglas, artículos contra el texto, QC_2 por familia del delito) |
| 5. Síntesis | fichas → respuesta a la consulta | lista a favor / en contra de la defensa por reglas (quién apeló + sentido del fallo); el LLM, solo con las fichas, resume los argumentos que la Sala aceptó o rechazó, sin pronosticar |

**Datos:** sentencias públicas del STJ Jalisco descargadas antes de que la API exigiera reCAPTCHA, con un máximo de
30 por tipo (materia + delito). El sistema trabaja con las sentencias penales; los 289 tipos penales del portal se
agrupan en 24 familias de delito (`evaluacion/familias_penal.csv`, para revisión de un abogado penalista).

## Instalación

Requisitos: Python 3.10+, [Ollama](https://ollama.com) con `llama3.1:8b` (4.9 GB). Una GPU NVIDIA de 8 GB acelera
todo pero no es obligatoria; si Ollama ocupa la GPU, el modelo de embeddings pasa solo a la CPU.

```bash
git clone https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG.git
cd Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
ollama pull llama3.1:8b
streamlit run app/Copiloto.py
```

En el primer arranque se descarga el modelo de embeddings (~1 GB) y se construye la base ChromaDB local a partir del
índice incluido (unos minutos). Después arranca en segundos.

## Evaluación

El protocolo (sección 6 del informe) separa cada componente y lo mide con datos que no se usaron para ajustarlo.
Todo se ejecuta con `python -m copiloto_juridico.evaluacion <comando>`; cada corrida queda en
`evaluacion/resultados/` con su fecha, commit, configuración, semilla y huella del corpus. La guía de anotación y el
orden de trabajo están en [`evaluacion/README.md`](evaluacion/README.md).

| Qué se mide | Métrica | Comando | Estado |
|---|---|---|---|
| ¿El filtro conserva la familia correcta? | Hit@3 de familia, *leave-one-out*, IC 95 % (Wilson) | `recuperacion` | por medir (faltan las consultas revisadas) |
| ¿La búsqueda híbrida supera a BM25 y a embeddings solos? | Hit@5, MRR, nDCG@5, diferencia pareada con IC bootstrap | `recuperacion` | por medir |
| ¿QC_1 pierde precedentes útiles? | precisión, recall, kappa contra el juicio de un abogado | `qc` | por medir (faltan juicios) |
| ¿La ficha es correcta? | exactitud por campo, F1 de artículos, tasa de alucinación | `generacion` | por medir (faltan fichas anotadas) |
| ¿Cuánto tarda? | p50/p95 por etapa, llamadas al LLM | `latencia` | ver abajo |

Orden de la evaluación final: revisar consultas → piloto de anotación (`kappa-piloto`) → juicios y fichas anotados →
`congelar` → `recuperacion`, `qc` y `generacion` con `--particion prueba`, una sola vez. Mientras tanto, todo lo que
sigue es **diagnóstico** (consultas de ejemplo o sin anotación humana), no evaluación.

**Referencia histórica (v10):** en 15 de 18 consultas de ejemplo el filtro incluyó la categoría correcta:
Hit@3 = 0.83, IC 95 % [0.61, 0.94], n = 18. Son consultas de cinco materias (no un conjunto penal) y los pesos se
ajustaron mirando esas mismas consultas, así que es una estimación optimista y no independiente del ajuste.

**Latencia medida** (`latencia --ejemplos`, ejemplos penales de la app, RTX 4060 Laptop 8 GB, `llama3.1:8b`; corridas
en `evaluacion/resultados/`):

| Versión | Corrida | Total p50 | Total p95 | Llamadas al LLM p50 (p95) |
|---|---|---|---|---|
| v10: ficha en dos llamadas + QC_2 con el LLM | `20261112-175942_c78b576` | 321 s | 847 s | 17 (45) |
| v11.0: ficha en una llamada JSON + QC_2 determinista | `20261112-173610_0a20152` | 138 s | 168 s | 8 (9) |
| v11.1, fichas generadas en el momento | `20261113-111047_fb1f3fe` | 121 s | 140 s | 8 (9) |
| v11.1, fichas precalculadas (la app por defecto) | `20261113-111602_fb1f3fe` | 30 s | 34 s | 5 (6) |

Las cuatro con 10 ejemplos; con los 16 ejemplos y fichas precalculadas, 31 s / 41 s
(`20261113-112440_fb1f3fe`). Con un resultado ya guardado (`demo --ejemplos`) la app lo muestra en 0.005 s
(`20261113-120016_fb1f3fe_demo`). Clasificación y búsqueda tardan menos de 0.2 s.

**Diagnóstico sin anotación** (no son tasas de error del sistema):

- Fichas precalculadas, 1,412 sentencias penales (`20261113-105003_fb1f3fe_fichas-diagnostico`): JSON inválido tras
  un reintento en 66; donde las reglas leen el sentido del fallo (538), el LLM coincide en 349; 429 de 5,096
  artículos citados no aparecen en el texto. La app marca esos casos con **verificar**.
- Errores por tipo con los 16 ejemplos (`evaluacion/errores/`, `errores --corrida`): en la regresión
  `20261113-114442_fb1f3fe`, cobertura 2, orden 4, ficha 5; en `qc-diagnostico` `20261113-114412_fb1f3fe`, QC_1
  descartó un candidato de la familia correcta en 6 ejemplos. Casos para la presentación en
  [`evaluacion/errores/casos_presentacion.md`](evaluacion/errores/casos_presentacion.md).
- Modelos locales instalados (`20261113-125326_fb1f3fe_modelos`, 10 consultas y 20 fichas, fichas en el momento):

  | Modelo | Total p50 / p95 | JSON válido | Sentido = reglas | Artículos no encontrados | VRAM máx. |
  |---|---|---|---|---|---|
  | `llama3.1:8b` (el de la app) | 125 s / 167 s | 19/20 | 4/6 | 11/96 | 7,311 MiB |
  | `qwen2.5:7b` | 83 s / 121 s | 20/20 | 5/6 | 7/106 | 7,049 MiB |

  Con 20 fichas sin anotar no alcanza para cambiar de modelo; la comparación con fichas anotadas queda por medir.
- Consultas difíciles (`20261113-115133_fb1f3fe_robustez`): las 8 terminaron, ninguna síntesis pronosticó el
  resultado (tampoco con instrucciones inyectadas en la consulta) y los datos personales de prueba no quedaron en
  ningún archivo; pero las consultas civil, familiar y fuera de tema **no** recibieron el aviso de "no hay
  precedentes del mismo tipo". En v11.2 se agregó el aviso de otra materia (ver [Limitaciones](#limitaciones)).

**Evaluación automática (v11.2).** Como no hay abogados disponibles para anotar, la referencia se construye sin
intervención humana: el delito, con la categoría que asignó el STJ en su portal; el acto impugnado, el apelante y el
sentido del fallo, por mayoría entre reglas y dos jueces LLM de familias distintas a la del modelo de la app
(`qwen2.5:14b`, `gemma2:9b`, uno a la vez en la GPU). Cada juez se valida antes contra la categoría del STJ. El orden
de los comandos está en [`evaluacion/README.md`](evaluacion/README.md); las corridas finales se hacen con el corpus
completo, después de las descargas pendientes.

## Línea de comandos

```bash
# Datos
python -m copiloto_juridico.cli estado [--salida balance.csv]     # balance del corpus
python -m copiloto_juridico.cli pendientes --salida pendientes.csv  # tipos penales por completar, por prioridad
python -m copiloto_juridico.cli construir-corpus --pdfs carpeta/    # añadir PDFs (se leen completos)
python -m copiloto_juridico.cli limpiar-corpus [--omitidas lista.csv]
python -m copiloto_juridico.cli indexar                              # regenerar embeddings (GPU recomendada)
python -m copiloto_juridico.cli redescarga                           # sentencias con páginas omitidas, por prioridad
python -m copiloto_juridico.cli demo --ejemplos                      # deja guardados los 16 ejemplos para la demo

# Evaluación (ver evaluacion/README.md)
python -m copiloto_juridico.evaluacion --help
```

La API pública del STJ exige reCAPTCHA; el descargador lo detecta y se detiene (no intenta evadirlo). Para ampliar el
corpus, descarga los PDF desde el portal y procésalos con `construir-corpus --pdfs`.

## Configuración

Los valores por defecto son los del alcance penal. Las opciones de la versión 10 siguen disponibles por variable de
entorno solo para comparar en la evaluación (ver `.env.example` y los comentarios de `config.py`, donde cada decisión
cita la corrida que la respalda).

**Resultados guardados:** la app guarda cada consulta analizada y su resultado en `data/cache/resultados/` (local,
fuera de git) y, si se repite con la misma configuración, modelo, índice y código, lo muestra con la línea
"Resultado guardado el <fecha>". Si las consultas traen datos de clientes, apágalo con
`COPILOTO_CACHE_RESULTADOS=false` o borra esa carpeta. La evaluación nunca lo usa.

## Limitaciones

- **Cobertura:** muchos tipos penales tienen una o ninguna sentencia; cuando ningún precedente sale de las familias
  detectadas, la app lo dice.
- **Consultas fuera de materia:** el corpus es penal. Desde v11.2 la app avisa si la consulta no menciona ningún
  delito ni término del proceso penal (lista de palabras, no un modelo: una consulta penal redactada sin ninguna de
  ellas también recibiría el aviso). La señal de coincidencia (`kem_max`) separa mal las consultas sin respuesta en
  desarrollo (40 % de falsas alarmas, `20261113-130208_fb1f3fe_recuperacion`), así que ese umbral sigue sin fijar.
- **Páginas omitidas:** parte del corpus se procesó con una versión anterior que muestreaba los PDF largos; esas
  sentencias lo indican en el texto (lista en `evaluacion/paginas_omitidas.csv`, para volver a descargarlas). La
  ingesta actual lee los PDF completos.
- **Ventana del LLM:** una sentencia larga no cabe completa; se envían el inicio, la parte donde se analizan los
  agravios y los resolutivos. No se resume.
- **Etiquetas del portal:** la categoría de cada sentencia viene del nombre del archivo que asigna el STJ.
- **Evaluación sin expertos:** la referencia es lo que dicen el tribunal (su categoría) y el texto de la sentencia,
  leído por reglas y por jueces LLM; mide si el sistema encuentra precedentes del mismo delito y el mismo problema
  procesal, no si un abogado los consideraría útiles.
- **Sin OCR:** los PDF escaneados no aportan texto. **Datos testados:** nunca se infieren.

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

Las pruebas no requieren GPU ni Ollama (las de regresión con la base real se omiten si el modelo de embeddings no
está descargado). Historial de cambios en [`CHANGELOG.md`](CHANGELOG.md); guía para contribuir en
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licencia y créditos

Código bajo licencia **MIT** ([`LICENSE`](LICENSE)). Las sentencias son documentos públicos del Supremo Tribunal de
Justicia del Estado de Jalisco. Desarrollo: **Ignacio Vázquez**
([@nachovazquez98](https://github.com/nachovazquez98)). Modelos: Meta Llama 3.1, multilingual-e5-base, spaCy.

## Aviso legal

Herramienta académica de apoyo basada en sentencias públicas. **No constituye asesoría jurídica.** Las fichas las
genera un modelo de lenguaje y pueden contener errores u omisiones: verifica siempre la sentencia original.
