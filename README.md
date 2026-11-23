<div align="center">

# ⚖️ Co-piloto Jurídico

**Búsqueda y análisis de precedentes penales de apelación del Supremo Tribunal de Justicia de Jalisco para tres
perfiles de usuario —el abogado (defensor o asesor jurídico de la víctima), la persona justiciable y el tribunal de
alzada—, con búsqueda híbrida (embeddings + BM25), un LLM local y una interfaz web.**

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
leer cada resolución completa. Ese costo de localización y lectura lo enfrentan por igual quienes litigan, quienes
son parte de un proceso y el propio tribunal.

**Usuarios destinatarios.** El sistema atiende tres perfiles, seleccionables en la interfaz; los tres usan la misma
búsqueda y las mismas fichas, y cada uno tiene sus propios ejemplos:

| Perfil | Necesidad | Uso del sistema |
|---|---|---|
| **Abogado** (defensor o asesor jurídico de la víctima, público o particular) — usuario principal | Interponer o contestar un recurso de apelación ante las Salas Penales | Precedentes en que la Sala calificó agravios semejantes como fundados, infundados o inoperantes |
| **Persona justiciable** (víctima, ofendido, imputado o familiar, sin formación jurídica) | Conocer el criterio con que el tribunal ha resuelto asuntos semejantes al suyo | Fichas con la parte favorecida y el sentido del fallo; la app remite a la asesoría jurídica para actuar |
| **Tribunal de alzada** (Sala Penal) | Resolver con un criterio congruente con sus propias resoluciones | Contraste del criterio de las Salas en supuestos semejantes; el sentido de la resolución corresponde exclusivamente a la Sala |

**Objeto:** a partir de la descripción de un caso en lenguaje natural, recuperar las resoluciones penales del propio
tribunal más afines y elaborar una ficha de cada una (delito, acto impugnado, agravios y su calificación por la Sala,
sentido del fallo, fundamento legal), así como una síntesis comparativa.

**Alcance:** el sistema no pronostica el resultado del asunto, no redacta agravios y no sustituye la lectura de la
resolución original ni el criterio profesional. Las resoluciones del STJ constituyen precedentes orientadores, no
jurisprudencia obligatoria.

## Uso en 4 pasos

1. Instala y abre la app (ver [Instalación](#instalación)): `streamlit run app/Copiloto.py`.
2. Elige el **perfil de usuario** y describe el caso (hechos, delito, resolución apelada y agravios) o carga un
   ejemplo de ese perfil (**Otro ejemplo** elige uno al azar).
3. Pulsa **Buscar precedentes** y espera a que terminen los 5 pasos.
4. Lee la síntesis (empieza con la parte que favoreció cada precedente) y las tarjetas.
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
| 3. Búsqueda | familias (o los delitos que elige el usuario, uno o varios) → hasta 30 sentencias candidatas | embeddings `multilingual-e5-base` (ChromaDB) + BM25, fusión RRF; filtro por familia con respaldo sin filtro |
| 4. Fichas | candidato → ficha penal | QC_1 (LLM: ¿sirve como precedente?), ficha en **una** llamada (JSON validado), verificaciones deterministas (sentido del fallo por reglas, artículos contra el texto, QC_2 por familia del delito) |
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

Orden de la evaluación final: corpus definitivo → fichas → juez y `referencias` → `congelar` → corridas con
`--particion prueba`, una sola vez. Lo que sigue en esta sección es **diagnóstico** de versiones anteriores.

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
| v11.2, solo penal, fichas precalculadas | `20261121-052935_4b1529e` | 31 s | 53 s | 5 (6) |

Las cuatro con 10 ejemplos; con los 16 ejemplos y fichas precalculadas, 31 s / 41 s
(`20261113-112440_fb1f3fe`). Con un resultado ya guardado (`demo --ejemplos`) la app lo muestra en 0.005 s
(`20261113-120016_fb1f3fe_demo`; con el corpus final, DEMO_FINAL). Clasificación y búsqueda tardan menos de
0.2 s.

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

  Con 20 fichas no alcanza para cambiar de modelo; `qwen2.5:7b` se usa como modelo de respaldo de las fichas precalculadas.
- Consultas difíciles (`20261113-115133_fb1f3fe_robustez`): las 8 terminaron, ninguna síntesis pronosticó el
  resultado (tampoco con instrucciones inyectadas en la consulta) y los datos personales de prueba no quedaron en
  ningún archivo; pero las consultas civil, familiar y fuera de tema **no** recibieron el aviso de "no hay
  precedentes del mismo tipo". En v11.2 se agregó el aviso de otra materia, que aparece en las 3
  (`20261121-053544_4b1529e_robustez`).

**Evaluación automática (v11.2).** El delito sale de la categoría que asignó el STJ en su portal; el acto
impugnado, el apelante y el sentido del fallo, de las reglas sobre el texto: el juez `gemma2:9b` acertó la familia del
STJ en 0.742 de 725 sentencias del corpus final, por debajo del mínimo de 0.80 fijado antes de correrlo, y no se usa
(`20261121-031724_723fe6b_referencias`, análisis en `evaluacion/errores/juez_familia.md`); `mistral:7b` tampoco
alcanzó el mínimo (`20261117-065346_a60ca4f_referencias`). El juez trabaja sobre una muestra estable (30 % de las sentencias, 50 % de
las consultas con todos sus pares).

## Línea de comandos

```bash
# Datos
python -m copiloto_juridico.cli estado [--salida balance.csv]     # balance del corpus
python -m copiloto_juridico.cli pendientes --salida pendientes.csv  # tipos penales por completar, por prioridad
python -m copiloto_juridico.cli construir-corpus --pdfs carpeta/    # añadir PDFs (se leen completos)
python -m copiloto_juridico.cli limpiar-corpus [--omitidas lista.csv]
python -m copiloto_juridico.cli indexar                              # regenerar embeddings (GPU recomendada)
python -m copiloto_juridico.cli redescarga                           # sentencias con páginas omitidas, por prioridad
python -m copiloto_juridico.cli demo --ejemplos                      # deja guardados los ejemplos de los tres perfiles

# Evaluación (ver evaluacion/README.md)
python -m copiloto_juridico.evaluacion --help
```

La API pública del STJ exige reCAPTCHA; el descargador lo detecta y se detiene (no intenta evadirlo). Para ampliar el
corpus, descarga los PDF desde el portal y procésalos con `construir-corpus --pdfs`.

## Configuración

Los valores por defecto son los del alcance penal. Las opciones de la versión 10 siguen disponibles por variable de
entorno solo para comparar en la evaluación (ver `.env.example` y los comentarios de `config.py`, donde cada decisión
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
- **Perfiles de usuario:** la evaluación mide la búsqueda y las fichas, que son las mismas para los tres perfiles;
  las consultas de evaluación están redactadas como las plantearía un abogado, así que no hay todavía una medición
  propia con consultas en lenguaje llano ni con el planteamiento de un toca en estudio.
- **Evaluación sin expertos:** la referencia es lo que dicen el tribunal (su categoría) y el texto de la sentencia,
  leído por reglas y por jueces LLM; mide si el sistema encuentra precedentes del mismo delito y el mismo problema
  procesal, no si un abogado los consideraría útiles.
- **Sin OCR:** los PDF escaneados no aportan texto. **Datos testados:** nunca se infieren.

## Próximos pasos

Continuación, mejora y evolución del proyecto, en orden de prioridad:

1. **Precisión del fundamento legal.** PASO_FUNDAMENTO
2. **Evaluación por perfil de usuario.** Conjuntos de consultas propios de la persona justiciable (lenguaje llano)
   y del tribunal de alzada (toca en estudio), con la misma referencia automática y una partición de prueba única.
3. **Tribunal de alzada.** Integrar la búsqueda por hechos y la ficha en el portal del STJ, lo que resolvería el
   problema de localización en su origen para los tres perfiles; normalizar el catálogo del portal con las 25
   familias de delito (hoy 546 tipos) y señalar resoluciones de distintas Salas con sentido opuesto en supuestos
   semejantes.
4. **Persona justiciable.** Ficha en lenguaje llano con el glosario de la app y canalización a la asesoría jurídica
   gratuita (defensoría pública o asesoría jurídica de víctimas).
5. **Corpus.** Ampliarlo con el procedimiento de [`descarga_datos/`](descarga_datos/) (una persona resuelve el
   reCAPTCHA), volver a descargar las sentencias con páginas omitidas y agregar OCR para los PDF escaneados.
6. **Controles.** Mejorar la precisión de QC_1 (0.322 en prueba) sin perder exhaustividad y fijar el umbral de
   «sin precedentes del mismo tipo» cuando una señal lo separe mejor que `kem_max`.
7. **Validación con especialistas.** Cuando haya abogados disponibles, contrastar la referencia automática con su
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
