<div align="center">

# ⚖️ Co-piloto Jurídico

**Búsqueda y análisis de precedentes penales de apelación del Supremo Tribunal de Justicia de Jalisco para el
abogado (defensor o asesor jurídico de la víctima) y el tribunal de alzada, con búsqueda híbrida (embeddings + BM25),
un LLM local y una interfaz web.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App_web-ff4b4b?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-LLM_local-blueviolet)
![Version](https://img.shields.io/badge/versión-11.2.0-informational)
![License](https://img.shields.io/badge/licencia-MIT-green)

</div>

---

## Objeto y alcance

**Problema:** el portal público del Supremo Tribunal de Justicia del Estado de Jalisco (STJ) localiza las sentencias
por categoría del delito, número de toca o fecha; conocer los hechos, los agravios y cómo los calificó la Sala exige
leer cada resolución completa. Ese costo de localización y lectura lo enfrentan por igual quien litiga el recurso y
el propio tribunal que lo resuelve.

**Usuarios destinatarios.** El sistema atiende dos perfiles, seleccionables en la interfaz; ambos usan la misma
búsqueda y las mismas fichas, y cada uno tiene sus propios ejemplos:

| Perfil | Necesidad | Uso del sistema |
|---|---|---|
| **Abogado** (defensor o asesor jurídico de la víctima, público o particular) — usuario principal | Interponer o contestar un recurso de apelación ante las Salas Penales | Precedentes en que la Sala calificó agravios semejantes como fundados, infundados o inoperantes |
| **Tribunal de alzada** (Sala Penal) | Resolver con un criterio congruente con sus propias resoluciones | Contraste del criterio de las Salas en supuestos semejantes; el sentido de la resolución corresponde exclusivamente a la Sala |

**Objeto:** a partir de la descripción de un caso en lenguaje natural, recuperar las resoluciones penales del propio
tribunal más afines y elaborar una ficha de cada una (delito, acto impugnado, agravios y su calificación por la Sala,
sentido del fallo, fundamento legal), así como una síntesis comparativa.

**Alcance:** la consulta de una persona sin formación jurídica (víctima, ofendido, imputado o familiar) queda fuera
del alcance de esta versión: no tiene perfil en la interfaz ni se ha evaluado. El sistema no pronostica el resultado del asunto, no redacta agravios y no sustituye la lectura de la
resolución original ni el criterio profesional. Las resoluciones del STJ constituyen precedentes orientadores, no
jurisprudencia obligatoria.

## Uso en 4 pasos

1. Instala y abre la app (ver [Instalación](#instalación)): `streamlit run app/Copiloto.py`.
2. Elige el **perfil de usuario** y describe el caso (hechos, delito, resolución apelada y agravios) o carga un
   ejemplo de ese perfil (**Otro ejemplo** elige uno al azar).
3. Pulsa **Buscar precedentes** y espera a que terminen los 5 pasos.
4. Lee la síntesis (empieza con la parte que favoreció cada precedente) y las tarjetas.
   Donde aparezca **verificar** (el sentido del fallo de la ficha no coincide con los resolutivos, o un artículo
   citado no está en la sentencia) o **sin verificar** (el sentido es el de la ficha porque no se encontraron los
   puntos resolutivos), revisa la sentencia original antes de citarla.
   Puedes descargar el informe en Markdown.

La página **Cómo funciona** explica el flujo, los límites de tamaño, un glosario jurídico breve y muestra la última
evaluación guardada.

## Cómo funciona (resumen)

| Paso | Entrada → salida | Técnica |
|---|---|---|
| 1. Conceptos clave | consulta → delito + problema procesal | LLM (Ollama, `llama3.1:8b`) |
| 2. Clasificación | consulta → 3 familias de delito más probables | coseno de embeddings + Jaccard de lemas + voto de las sentencias recuperadas |
| 3. Búsqueda | familias (o los delitos que elige el usuario, uno o varios) → hasta 30 sentencias candidatas | embeddings `multilingual-e5-base` (ChromaDB) + BM25, fusión RRF; filtro por familia con respaldo sin filtro |
| 4. Fichas | candidato → ficha penal | orden por familia y acto impugnado; QC_1 opcional (LLM: ¿sirve como precedente?, apagado por defecto), ficha en **una** llamada (JSON validado), verificaciones deterministas (sentido del fallo por reglas, artículos contra el texto, QC_2 por familia del delito) |
| 5. Síntesis | fichas → respuesta a la consulta | parte favorecida por cada precedente, por reglas (quién apeló + sentido del fallo); el LLM, solo con las fichas, resume los argumentos que la Sala aceptó o rechazó, sin pronosticar |

**Datos:** sentencias públicas del STJ Jalisco, con un máximo de 30 por tipo (materia + delito): las descargadas
de la API antes de que exigiera reCAPTCHA y las que se bajaron después a mano desde el portal (`construir-corpus
--pdfs` las valida). El corpus contiene solo resoluciones penales: 2,495 sentencias, cuyos 546 tipos del portal se
agrupan en 25 familias de delito, incluida «cuestiones procesales» para quejas, excusas y conflictos de competencia,
que no son delitos (`evaluacion/familias_penal.csv`, asignadas por reglas; ninguna revisada por un especialista).

**Qué incluye el repositorio:** el texto depurado de las 2,495 sentencias (`data/corpus/`), sus embeddings
precalculados (`data/indice/`: 186,971 fragmentos × 768, float16, en archivos de 23 MB), las fichas precalculadas
(`data/fichas/`), las corridas de la evaluación (`evaluacion/resultados/`), el notebook técnico (`notebooks/`) y la
presentación al jurado con sus notas del orador (`docs/v11.2 Co-piloto Jurídico - Presentación.pdf` y
`... - Notas del orador.pdf`). Los 1,899 PDF originales descargados a mano (1.3 GB) no van en git: están como
archivos adjuntos de la versión publicada [`v11.2`](https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG/releases/tag/v11.2)
en 15 ZIP de menos de 90 MB (`sentencias-penales-pdf-01-de-15.zip` … `-15-de-15.zip`, carpetas `nuevas/` y
`redescarga/`); se descomprimen juntos en `data/raw/` solo para reconstruir el corpus.

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

Para usar la app basta con clonar: los PDF originales no están en git y `git clone` no los descarga. Solo se
necesitan para reconstruir el corpus; en ese caso se descargan aparte, desde la versión publicada `v11.2`:

```bash
for i in $(seq -w 1 15); do
  curl -LO https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG/releases/download/v11.2/sentencias-penales-pdf-$i-de-15.zip
done
for f in sentencias-penales-pdf-*-de-15.zip; do unzip -q -o "$f" -d data/raw/; done
```

Opcional, solo para construir el corpus con PDF escaneados: `tesseract` y `pdftoppm` (poppler) con el modelo de
español `spa` (por ejemplo `tesseract-langpack-spa` o `tesseract-ocr-spa` en Linux, o `spa.traineddata` en la carpeta que
indique `TESSDATA_PREFIX`). Sin ellos, esas páginas se omiten (`COPILOTO_PDF_OCR=false` lo desactiva).

En el primer arranque se descarga el modelo de embeddings (~1 GB), se construye la base ChromaDB local a partir del
índice incluido (unos minutos) y se guarda el índice BM25 en `data/cache/`. Después la app abre en unos 8 s (medido con
el corpus final: 22.5 s si tuviera que reconstruir BM25 en cada arranque); las consultas de los ejemplos ya
analizadas (`python -m copiloto_juridico.cli demo --ejemplos`) se muestran al instante.

## Evaluación

El protocolo separa cada componente y lo mide con datos que no se usaron para ajustarlo. Todo se ejecuta con
`python -m copiloto_juridico.evaluacion <comando>`; cada corrida queda en `evaluacion/resultados/` con su fecha,
commit, configuración, semilla y huella del corpus. No hay abogados disponibles: la referencia es automática (la
categoría del STJ, reglas y un juez LLM validado) y se describe en [`evaluacion/README.md`](evaluacion/README.md).

Resultados de la evaluación final: partición de prueba, una sola corrida por comando con el congelado
`a15b83c852281010` (commit `4b1529e`); 46 consultas validadas (más 2 sin respuesta en el corpus), IC 95 %.

| Qué se mide | Métrica | Resultado | Corrida |
|---|---|---|---|
| ¿El filtro conserva la familia correcta? | Hit@3 de familia (Wilson) | **0.717** [0.575, 0.827]; línea base k-NN 0.435; diferencia +0.283 [+0.130, +0.435], significativa | `20261121-032545_4b1529e_recuperacion` |
| ¿La búsqueda híbrida supera a BM25 y a embeddings solos? | nDCG@5 contra la referencia, diferencia pareada (bootstrap) | app 0.285 [0.211, 0.355]; BM25 0.243; embeddings 0.304; diferencias no significativas con n = 46. Con metadatos, RRF supera a embeddings: +0.093 [+0.042, +0.143] | `20261121-032545_4b1529e_recuperacion` |
| ¿Conviene el re-ranker? | criterio fijado: nDCG@5 contra la referencia | Hit@5 del documento fuente +0.196 [+0.065, +0.326]; nDCG@5 −0.029 [−0.094, +0.039]: no cumple el criterio y queda apagado | `20261121-032545_4b1529e_recuperacion` |
| ¿QC_1 pierde precedentes útiles? | recall, precisión, kappa (776 pares) | recall **0.721** [0.660, 0.775]; precisión 0.322; kappa 0.051 | `20261121-052256_4b1529e_qc` |
| ¿La ficha es correcta? | exactitud por campo (Wilson) | familia **0.917** [0.884, 0.942] (n = 350); acto impugnado 0.883; apelante 0.830; sentido del fallo 0.722 [0.665, 0.772] | `20261121-032553_4b1529e_fichas-referencia` |
| ¿El fundamento legal de la ficha es el de la Sala? | precisión y exhaustividad de los artículos (bootstrap) | precisión **0.644** [0.600, 0.690]; exhaustividad 0.133 (la ficha resume los principales); 48.2 % de las fichas con todas sus citas en el fundamento; 7 de 350 fichas citan algún artículo que no está en la sentencia | `20261121-032553_4b1529e_fichas-referencia` |
| ¿Ahorra trabajo? | sentencias que hay que abrir para encontrar 3 útiles (9 consultas) | con el delito elegido, mediana 22 frente a 31 del portal; elegir el delito ahorra 7.0 [+1.8, +13.2], significativo | `20261121-032625_4b1529e_esfuerzo` |
| ¿Cuánto tarda? | p50/p95 por consulta, llamadas al LLM | 30.8 s / 52.6 s; 5 llamadas (p50) | `20261121-052935_4b1529e_latencia` |
| ¿Resiste consultas difíciles? | 8 consultas de robustez | 8/8 terminan; aviso de otra materia en las 3 que lo requieren; ninguna síntesis pronostica el resultado; ningún archivo guarda los datos personales de prueba | `20261121-053544_4b1529e_robustez` |

Análisis de los fallos por tipo en `evaluacion/errores/` (`errores --corrida`). Guía para leer cada métrica en
[`evaluacion/README.md`](evaluacion/README.md).

**Mejoras probadas después de la evaluación final.** En la partición de desarrollo, por pares (IC 95 % por
bootstrap), sin volver a medir la de prueba. Se adoptó solo lo que mejoró sin empeorar otra medida:

| Cambio | Resultado | Decisión | Corrida |
|---|---|---|---|
| Síntesis con un ejemplo de líneas citadas y revisión final de las citas (19 consultas) | con un ejemplo de contenido jurídico, cita válida −0.024 [−0.144, +0.078], lo copia en 9 de 19 síntesis y pronostica en una (una primera medición, con otros precedentes, daba +0.086); con un ejemplo solo de formato, +0.024 [−0.069, +0.121], sin copias | descartado: la app sigue con la síntesis anterior | `20261124-003427_905c255_sintesis-citas` |
| Precedentes que ve el usuario (39 consultas): la lista real de la app, con el orden por familia y acto impugnado, frente a la que medía `recuperacion` | P@3 0.248 → **0.410** (+0.162 [+0.077, +0.248]); condensada, sin los no juzgados, +0.128 [+0.043, +0.222]; nDCG@5 0.270 → 0.442 | medición: la calidad real es mayor que la reportada | `20261124-014247_0f2894a_precedentes` |
| QC_1 sobre los precedentes mostrados (39 consultas) | P@3 0.410 con y sin QC_1 (+0.000 [−0.034, +0.034]); nDCG@3 −0.007, no significativo; igual si la consulta nombra el acto o no. Sin QC_1, la consulta pasa de 30.8 s a 17.1 s (p50) y de 5 a 2 llamadas al modelo (`20261124-020004_2e50981_latencia`) | QC_1 apagado por defecto (opcional en Configuración) | `20261124-014247_0f2894a_precedentes` |
| QC_1 lee la ficha precalculada en lugar del fragmento (660 pares) | recall 0.725 → **0.844** (+0.120 [+0.036, +0.204]); especificidad 0.410 → 0.396 (no significativo); precisión 0.294 → 0.321 | adoptado | `20261123-230824_5fc9b67_qc` |
| QC_1 con `qwen2.5:7b` (660 pares) | recall 0.844 → 0.425 (−0.419, significativo); 1.9 s por llamada, pero cada cambio de modelo en 8 GB cuesta 5-8 s | descartado | `20261123-234717_e28a600_qc` |
| QC_1 solo con el veredicto, sin razonar (660 pares) | 0.59 s por llamada, pero responde SI en 649 de 660 pares (especificidad 0.022): deja de filtrar | descartado | `20261123-234717_e28a600_qc` |
| Ficha: pedir solo los artículos que invoca la Sala (120 sentencias) | precisión del fundamento +0.010 [−0.040, +0.060]; exhaustividad −0.023, significativa | descartado | `20261123-174502_b8fdd94_fundamento-prompt` |
| Ficha: darle al modelo la parte considerativa (120 sentencias) | precisión −0.048 [−0.096, +0.006]; exhaustividad −0.030, significativa | descartado | `20261123-212426_5fc9b67_fundamento-contexto` |
| Consultas planteadas como toca en estudio (39 consultas) | Hit@3 de familia igual (+0.051 [−0.077, +0.205]); nDCG@5 0.266 → 0.145, significativo | medición del perfil | `20261123-202205_e566e3d-sucio_perfil-tribunal` |

Orden de la evaluación final: corpus definitivo → fichas → `referencias` → `congelar` → corridas con
`--particion prueba`, una sola vez.

**Por qué una consulta tarda ~17 s** (`latencia --ejemplos`, 10 ejemplos penales de la app, RTX 4060 Laptop 8 GB,
`llama3.1:8b`):

| Diseño | Corrida | Total p50 | Total p95 | Llamadas al LLM p50 (p95) |
|---|---|---|---|---|
| Línea base: ficha en dos llamadas + QC_2 con el LLM | `20261112-175942_c78b576` | 321 s | 847 s | 17 (45) |
| Ficha en una llamada JSON + QC_2 determinista | `20261112-173610_0a20152` | 138 s | 168 s | 8 (9) |
| Además, fichas precalculadas, corpus final | `20261121-052935_4b1529e` | 31 s | 53 s | 5 (6) |
| Además, QC_1 apagado (la app) | `20261124-020004_2e50981` | **17 s** | **35 s** | 2 (2.6) |

Sin QC_1 queda la síntesis (15 s de mediana) y, si falta la ficha precalculada de un candidato, su generación;
clasificación y búsqueda tardan menos de 0.2 s. En los 20 ejemplos de la demo, apagar QC_1 bajó la mediana de 31.6 s
a 18.2 s (−11.8 s [−14.2, −9.4], 19 de 20 más rápidos; `20261124-005412_82c94ed_demo` frente a
`20261124-020755_2e50981_demo`), y un resultado ya guardado (`demo --ejemplos`) se muestra en 0.007 s.

**Diagnósticos complementarios** (sin referencia; no son tasas de error del sistema):

- Errores por tipo con los ejemplos (`evaluacion/errores/`, `errores --corrida`): en la regresión
  `20261113-114442_fb1f3fe`, cobertura 2, orden 4, ficha 5; en `qc-diagnostico` `20261113-114412_fb1f3fe`, QC_1
  descartó un candidato de la familia correcta en 6 ejemplos. Casos para la presentación en
  [`evaluacion/errores/casos_presentacion.md`](evaluacion/errores/casos_presentacion.md).
- Modelos locales instalados (`20261113-125326_fb1f3fe_modelos`, 10 consultas y 20 fichas, fichas en el momento):

  | Modelo | Total p50 / p95 | JSON válido | Sentido = reglas | Artículos no encontrados | VRAM máx. |
  |---|---|---|---|---|---|
  | `llama3.1:8b` (el de la app) | 125 s / 167 s | 19/20 | 4/6 | 11/96 | 7,311 MiB |
  | `qwen2.5:7b` | 83 s / 121 s | 20/20 | 5/6 | 7/106 | 7,049 MiB |

  Con 20 fichas no alcanza para cambiar de modelo; `qwen2.5:7b` se usa como modelo de respaldo de las fichas
  precalculadas; no como QC_1, donde pierde la mitad de los precedentes relevantes (tabla anterior).
- Consultas difíciles, con la versión final (`20261124-014756_b2da1e5_robustez`): las 8 terminan, ninguna síntesis pronostica el resultado (tampoco con
  instrucciones inyectadas en la consulta), las 3 de otra materia reciben el aviso y los datos personales de prueba
  no quedan en ningún archivo.

**Referencia automática.** El delito sale de la categoría que asignó el STJ en su portal; el acto impugnado, el
apelante y el sentido del fallo, de las reglas sobre el texto. El juez `gemma2:9b` acertó la familia del STJ en 0.742
de 725 sentencias del corpus final, por debajo del mínimo de 0.80 fijado antes de correrlo, y no se usa
(`20261121-031724_723fe6b_referencias`, análisis en `evaluacion/errores/juez_familia.md`); `mistral:7b` tampoco
alcanzó el mínimo (`20261117-065346_a60ca4f_referencias`). El juez trabaja sobre una muestra estable (30 % de las
sentencias, 50 % de las consultas con todos sus pares).

## Línea de comandos

```bash
# Datos
python -m copiloto_juridico.cli estado [--salida balance.csv]     # balance del corpus
python -m copiloto_juridico.cli pendientes --salida pendientes.csv  # tipos penales por completar, por prioridad
python -m copiloto_juridico.cli construir-corpus --pdfs carpeta/    # añadir PDFs (se leen completos)
python -m copiloto_juridico.cli limpiar-corpus [--omitidas lista.csv]
python -m copiloto_juridico.cli indexar                              # regenerar embeddings (GPU recomendada)
python -m copiloto_juridico.cli redescarga                           # sentencias con páginas omitidas, por prioridad
python -m copiloto_juridico.cli demo --ejemplos                      # deja guardados los ejemplos de los dos perfiles

# Evaluación (ver evaluacion/README.md)
python -m copiloto_juridico.evaluacion --help
```

**Descarga de sentencias de cualquier materia:** [`descarga_datos/descargar.py`](descarga_datos/) recorre el portal
del STJ con filtros por materia y delito, tope por tipo, validación de cada PDF y reanudación. Desde octubre de 2026 la
API exige reCAPTCHA: en el modo `navegador` una persona lo resuelve y el script continúa con esa sesión (nunca lo
resuelve ni lo evita). Los PDF se incorporan con `construir-corpus --pdfs`.

## Configuración

Los valores por defecto son los del alcance penal. La línea base con que la evaluación compara cada decisión está
disponible por variable de entorno (ver `.env.example` y los comentarios de `config.py`, donde cada decisión
cita la corrida que la respalda).

**Modelo de lenguaje:** `llama3.1:8b`, el evaluado. Cabe íntegro en 8 GB de GPU con la ventana de 8,192 tokens cuando
el modelo de embeddings se ejecuta en CPU; si este ocupa la GPU primero, Ollama deja una parte de las capas en la RAM.
`gemma2:9b` y `qwen2.5:14b` no caben. `qwen2.5:7b` también cabe y fue más rápido en el diagnóstico de modelos
(`20261113-125326_fb1f3fe_modelos`, 20 fichas sin referencia), pero como juez no alcanzó 0.80 al leer la familia del
delito, y cambiarlo invalida las 2,495 fichas precalculadas y la evaluación. Se usa como **modelo de respaldo de las
fichas** (`FICHAS_MODELO_RESPALDO`): `fichas-batch --respaldo` precalcula con él las fichas que faltan del modelo
principal, fuera de la muestra de la referencia, y la app las usa en lugar de generarlas durante la consulta. También
puede elegirse en la barra lateral como modelo completo, no evaluado. El análisis completo y el protocolo para adoptarlo
están en el notebook (`notebooks/`, sección 2.1).

**Resultados guardados:** la app guarda cada consulta analizada y su resultado en `data/cache/resultados/` (local,
fuera de git) y, si se repite con la misma configuración, modelo, índice y código, lo muestra con la línea
"Resultado guardado el <fecha>". Si las consultas traen datos de clientes, apágalo con
`COPILOTO_CACHE_RESULTADOS=false` o borra esa carpeta. La evaluación nunca lo usa.

## Limitaciones

- **Cobertura:** muchos tipos penales tienen una o ninguna sentencia; cuando ningún precedente sale de las familias
  detectadas, la app lo dice.
- **Consultas fuera de materia:** el corpus es penal. La app avisa si la consulta no menciona ningún
  delito ni término del proceso penal (lista de palabras, no un modelo: una consulta penal redactada sin ninguna de
  ellas también recibiría el aviso). La señal de coincidencia (`kem_max`) separa mal las consultas sin respuesta en
  desarrollo (40 % de falsas alarmas, `20261113-130208_fb1f3fe_recuperacion`), así que ese umbral sigue sin fijar.
- **Páginas omitidas:** parte del corpus se procesó con una versión anterior que muestreaba los PDF largos; esas
  sentencias lo indican en el texto (lista en `evaluacion/paginas_omitidas.csv`, para volver a descargarlas). La
  ingesta actual lee los PDF completos.
- **Ventana del LLM:** una sentencia larga no cabe completa; se envían el inicio, la parte donde se analizan los
  agravios y los resolutivos. No se resume.
- **Sentido del fallo:** el campo de la ficha acierta en 0.722 de los casos; por eso la tarjeta muestra el sentido
  leído de los puntos resolutivos cuando las reglas lo encuentran (1,966 de 2,495 sentencias,
  `20261120-014423_c27877e_metadatos-cobertura`; 53 de los 60 precedentes de la demo) y marca **sin verificar** el
  resto. Donde no lo encuentran suele tratarse de resoluciones sin sentido de apelación (excusas, competencia) o de
  textos con páginas omitidas.
- **Etiquetas del portal:** la categoría de cada sentencia viene del nombre del archivo que asigna el STJ.
- **Perfiles de usuario:** la búsqueda y las fichas son las mismas para los dos perfiles, y la evaluación final usa
  consultas redactadas como las plantearía un abogado. Con las mismas 39 consultas de desarrollo reescritas por el LLM
  como toca en estudio, la familia correcta se encuentra igual (Hit@3 0.667 → 0.718; +0.051 [−0.077, +0.205], no
  significativo), pero los cinco primeros precedentes son menos pertinentes (nDCG@5 0.266 → 0.145; −0.121
  [−0.205, −0.042], significativo; `20261123-202205_e566e3d-sucio_perfil-tribunal`). Parte de la diferencia puede venir
  de la reescritura, que a veces altera o generaliza la consulta; por eso la app pide al tribunal hechos, acto
  impugnado, apelante y agravios concretos.
- **Evaluación sin expertos:** la referencia es lo que dicen el tribunal (su categoría) y el texto de la sentencia,
  leído por reglas y por jueces LLM; mide si el sistema encuentra precedentes del mismo delito y el mismo problema
  procesal, no si un abogado los consideraría útiles.
- **Documentos escaneados:** `construir-corpus` lee por OCR (Tesseract, español) las páginas sin capa de texto si
  Tesseract está instalado. En los 1,899 PDF del proyecto solo uno está escaneado (94 páginas; el OCR lo lee en 392 s
  y las reglas encuentran su sentido y su fundamento) y otras 12 páginas sueltas no tienen texto. Ese PDF no se
  incorporó: no cambia el corpus congelado de la evaluación y su versión escaneada no tiene suprimidos los datos
  personales. **Datos testados:** nunca se infieren.

## Próximos pasos

Continuación, mejora y evolución del proyecto, en orden de prioridad:

1. **Precisión del fundamento legal.** Pedir en el prompt solo los artículos que la Sala invoca para fundar su
   resolución no la mejora: en 120 sentencias de desarrollo, por pares, la precisión pasa de 0.719 a 0.729
   (+0.010 [−0.040, +0.060], no significativa) y la exhaustividad baja de 0.146 a 0.123 (significativo); no se
   adoptó (`20261123-174502_b8fdd94_fundamento-prompt`). De 127 artículos que la ficha cita fuera del fundamento de
   la Sala, 126 sí aparecen en la sentencia: el error es de selección (agravios, primera instancia), no de invención.
   Tampoco la mejora darle al modelo la parte considerativa en lugar de la ventana de los agravios, aunque así ve más
   artículos del fundamento (80.3 % frente a 75.6 % en 297 sentencias): precisión 0.717 → 0.669 (−0.048
   [−0.096, +0.006]) y exhaustividad 0.147 → 0.116 (significativo), sin empeorar el sentido ni los agravios
   (`20261123-212426_5fc9b67_fundamento-contexto`). El límite está en cómo el modelo de 8B elige los artículos, no
   en lo que ve; queda para un modelo mayor, cuando quepa en el equipo, o para la validación con especialistas
   (punto 6), que permitiría medir estrategias no basadas en las mismas reglas de la referencia.
2. **Evaluación del perfil del tribunal de alzada.** La medición en desarrollo (ver Limitaciones) usa consultas
   reescritas por el LLM; el siguiente paso es un conjunto redactado por personal de una Sala, con la misma
   referencia automática y una partición de prueba única.
3. **Tribunal de alzada.** Integrar la búsqueda por hechos y la ficha en el portal del STJ, lo que resolvería el
   problema de localización en su origen; normalizar su catálogo (546 tipos) con la agrupación en 25 familias de
   [`evaluacion/familias_penal.csv`](evaluacion/familias_penal.csv), y señalar resoluciones de distintas Salas con
   sentido opuesto en supuestos semejantes. Esto último requiere leer la Sala y el sentido con más exactitud que la
   ficha actual (sentido del fallo 0.722 en prueba), para no señalar falsas divergencias.
4. **Corpus.** Ampliarlo con el procedimiento de [`descarga_datos/`](descarga_datos/) (una persona resuelve el
   reCAPTCHA) y volver a descargar las sentencias con páginas omitidas; los PDF escaneados ya se leen por OCR al
   construir el corpus, pero antes de incorporarlos hay que comprobar que tengan suprimidos los datos personales.
5. **Consultas sin acto impugnado.** Cuando la consulta no nombra la resolución que se apela, no se aplica el orden
   por familia y acto y la precisión de los precedentes mostrados cae (P@3 0.246 frente a 0.567 en desarrollo). La
   app ya pide nombrarla; el siguiente paso es deducir el acto de la descripción y medir el cambio en una partición de
   prueba nueva. También queda fijar el umbral de «sin precedentes del mismo tipo» cuando una señal lo separe mejor
   que `kem_max`.
6. **Validación con especialistas.** Cuando haya abogados disponibles, contrastar la referencia automática con su
   juicio sobre la utilidad de los precedentes.

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
