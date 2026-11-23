# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado [SemVer](https://semver.org/lang/es/).

## [11.2.0] - 2026-11-23

Evaluación sin anotadores humanos (no hay abogados disponibles): la referencia sale de la categoría que asignó el STJ,
del texto de la sentencia (reglas) y de un juez LLM de otra familia que la del modelo de la app, validado antes de
usarse. Corpus penal ampliado con descargas a mano desde el portal.

### Añadido
- Tres perfiles de usuario en la app, que resuelven el mismo problema de localización y lectura de precedentes: el
  abogado (defensor o asesor jurídico de la víctima), usuario principal; la persona justiciable y el tribunal de
  alzada. Cada perfil tiene sus ejemplos (`ejemplos.PERFILES`) y su advertencia de alcance; la búsqueda y las fichas
  son las mismas. En los ejemplos del abogado, su cliente es la víctima del delito (antes era el acusado que pedía
  quitarse los cargos). «Otro ejemplo» carga uno al azar del perfil elegido. Los ejemplos de defensa siguen en
  `ejemplos.EJEMPLOS_PENALES` porque la evaluación guardada se hizo con ellos.
- «Parte favorecida» (el acusado o la acusación) en la tarjeta, la ficha y el resultado de cada precedente de la
  síntesis, en lugar de «resultado para la defensa»; misma regla (apelante + sentido del fallo). La síntesis ya no se
  dirige solo al defensor.
- README: problema, perfiles de usuario y próximos pasos del proyecto.
- Índice BM25 guardado en `data/cache/` (se rehace solo si cambian el corpus o su código): la app abre en ~8 s en
  lugar de 22.5 s. Mismo resultado; cambia la huella del código, así que los ejemplos de la demo se volvieron a
  precalcular. Las corridas de prueba siguen atadas al congelado `a15b83c852281010`.
- Presentación al jurado y notas del orador en PDF (`docs/`); PDF de las sentencias como adjuntos de la versión `v11.2`, en 15 ZIP de menos de 90 MB.
- Evaluación final en la partición de prueba, una sola corrida por comando (congelado `a15b83c852281010`): Hit@3 de
  familia 0.717, exactitud de la familia en la ficha 0.917, precisión del fundamento legal 0.644, recall de QC_1 0.721,
  p50 de 30.8 s por consulta; tabla completa en el README y guía de lectura en `evaluacion/README.md`.
- `evaluacion jueces`: pares (consulta, sentencia) por agrupamiento y tandas de un juez a la vez, reanudables y con
  la clave del prompt, sobre una muestra estable por hash (30 % de las sentencias, 50 % de las consultas con todos sus
  pares; `--completa` para todo). Si el juez de relevancia omite la calificación, un tercer intento pide solo esa clave.
- `evaluacion referencias`: valida cada juez contra la categoría del STJ (mínimo 0.80 fijado antes de correrlos; con
  todas las sentencias si el juez las juzgó todas), referencia por mayoría de lectores para acto impugnado, apelante
  y sentido, juicios de relevancia y validación automática de las consultas.
- `evaluacion fichas-referencia`: fichas precalculadas contra la referencia, con IC 95 %.
- `evaluacion esfuerzo`: sentencias que hay que abrir para encontrar 3 útiles con el portal, la búsqueda por
  palabras y la app.
- `evaluacion familias --coherencia`: comprueba la agrupación tipo → familia con los embeddings del índice.
- `evaluacion borradores --ampliar N`: más consultas por familia sin tocar las existentes.
- `construir-corpus` rechaza un PDF con el nombre de una sentencia del corpus pero el texto de otra (se conserva el
  anterior; `data/corpus/descargas_rechazadas.csv`) y quita las sentencias publicadas dos veces con distinto nombre
  (`data/corpus/duplicados.csv`): contaban doble en la búsqueda y podían quedar en desarrollo y en prueba a la vez.
- `evaluacion/descargas/`: lista de las sentencias penales por descargar a mano (incompletas, nuevas y
  `3_volver_a_bajar.csv` con las que llegaron equivocadas).
- App: el usuario puede elegir, si quiere, uno o varios delitos (con cuántas sentencias hay de cada uno). La app
  sugiere los de la descripción sin llamar al LLM (botón "Usar la sugerencia"). Los elegidos sustituyen a las 3
  categorías del clasificador en el filtro, el orden por familia y QC_2; la búsqueda sin filtro sigue completando, y
  si la elección difiere de lo que habría buscado la app, lo dice. `esfuerzo` mide también "app con el delito
  elegido" frente a la app sola.

- `fichas-referencia`: precisión y exhaustividad del **fundamento legal** de la ficha frente a los artículos que la
  sentencia invoca como fundamento («con fundamento en», «de conformidad con», «con apoyo en», «en términos de», «con
  base en» y puntos resolutivos), leídos por reglas.
- Modelo de respaldo de las fichas (`FICHAS_MODELO_RESPALDO = qwen2.5:7b`, `fichas-batch --respaldo`): 181 fichas de
  sentencias sin ficha de llama3.1:8b, fuera de la muestra; la app las usa en lugar de generarlas en la consulta.
- `indexar --recortar`: reduce el índice al corpus sin recalcular embeddings.
- `evaluacion/errores/juez_familia.md` y `evaluacion/errores/sugerencia_delito.md`.

### Corregido
- Sentido del fallo por reglas: también "lo procedente es confirmar/revocar…" y el verbo sin "se" (cobertura 557 →
  608 de 1,412, sin cambiar ningún valor ya leído; `evaluacion/errores/sentido_fallo.md`).
- Artículos "no encontrados": la sentencia se lee de forma tolerante (listas con fracciones, "arábigos", "diversos",
  "en relación al", palabras partidas por el PDF): 429 → 23 de 5,096 (`evaluacion/errores/articulos_no_encontrados.md`).
- Ficha JSON: tercer intento que pide solo los campos obligatorios que faltan; las fichas inválidas se reintentan en
  `fichas-batch`, que además recalcula las verificaciones sin LLM.
- `construir-corpus` conserva las sentencias que ya estaban al aplicar el tope de 30 por tipo.
- Extracción de PDF: el inicio acepta un párrafo numerado ("1.VISTOS") y el corte final se busca después de los
  resolutivos. Un "Notifíquese" citado de primera instancia dejaba hasta el 1 % del documento (9 de 1,889 PDF
  conservaban menos del 30 %). 17 PDF cambian y todos conservan más texto. El PDF procesado reemplaza al texto
  guardado de la misma sentencia.
- `construir-corpus`: `descargas_rechazadas.csv` y `duplicados.csv` cubren todas las carpetas de la construcción
  (antes la segunda carpeta podía sobrescribir los rechazos de la primera).
- `fichas-batch`: si una sentencia deja al LLM sin respuesta en los 3 intentos, queda sin ficha y el lote sigue;
  si pasa con dos seguidas, el problema es el servidor y se detiene. `--muestra` genera solo las fichas de la
  muestra de la referencia.
- `evaluacion qc` medía QC_1 con el prompt v10; ahora usa el de la app (`prompt_qc1`, v11).
- `evaluacion jueces --pares` saca de la VRAM los modelos de Ollama antes de cargar el de embeddings (fallaba con
  `CUBLAS_STATUS_ALLOC_FAILED` en 8 GB).
- `referencias` armaba el prompt del juez con la lista de familias de la muestra y ninguna respuesta guardada
  coincidía; ahora la arma con todo el corpus.

### Cambiado
- Corpus: 357 sentencias penales vueltas a descargar completas, 51 nuevas y 22 repetidas fuera (11 penales; una
  consulta, A132, apunta ahora a la copia que se queda). Penal: 1,412 → 1,451 sentencias; con páginas omitidas,
  ~800 → 439. Sentido leído por reglas: 875 de 1,451 (`20261119-195104_3995979_metadatos-cobertura`).
- Corpus final (c27877e): con las listas 1 y 2 de `evaluacion/descargas/` y la extracción corregida, penal 2,495
  sentencias; índice de 211,572 fragmentos; 54 todavía con páginas omitidas (`evaluacion/paginas_omitidas.csv`).
  De los 56 PDF vueltos a bajar, 54 traían otra resolución del mismo día o del mismo toca, se rechazaron y conservan
  su texto anterior (`data/corpus/descargas_rechazadas.csv`). Sentido leído por reglas: 1,966 de 2,495
  (`20261120-014423_c27877e_metadatos-cobertura`).
- Jueces: `qwen2.5:7b` no pasó la prueba piloto, `qwen2.5:14b` no cabe en 8 GB, `mistral:7b` no alcanzó 0.80 (0.795
  en 1,412) y queda solo `gemma2:9b` (0.845 en la muestra de 401): `20261117-065346_a60ca4f_referencias`.
- La app avisa cuando la consulta no menciona ningún delito ni término del proceso penal (antes, las consultas
  civiles, familiares o ajenas al derecho recibían precedentes penales sin aviso).
- La página "Cómo funciona" muestra `fichas-referencia` y `esfuerzo` en lugar de `generacion`.
- Prueba de regresión v10: el caso civil "Daños y perjuicios (accidente vehicular)" queda registrado como fallo
  conocido. Con el corpus penal completo, "homicidio culposo" desplaza a "pago de daños y perjuicios" del top 3; es un
  cambio de datos, fuera del alcance penal.

- App en registro profesional: la página "Cómo funciona" documenta el método para el usuario (objeto y alcance,
  guía de uso, flujo, fundamentos de fragmentos, embeddings, ChromaDB, BM25, RRF, clasificación y controles,
  glosario procesal, evaluación y limitaciones); textos y avisos sin tuteo.
- Notebook reescrito como documentación técnica: llama al paquete en lugar de copiar código, con un diagrama por
  fase, el porqué de cada modelo y del equipo (con fuentes), el catálogo de prompts y la decisión sobre la materia
  civil; sin las versiones viejas del descargador.
- Sugerencia inicial de delito medida sin LLM: familia correcta en 73 de 100 consultas de desarrollo y 13 de 14
  ejemplos (`evaluacion/errores/sugerencia_delito.md`).

- **Solo materia penal:** fuera del corpus y del índice las 2,119 sentencias de civil, familiar, mercantil y laboral
  (186,971 de 211,572 fragmentos), los prompts de esas materias y las 12 consultas v10 de otras materias;
  `construir-corpus` guarda solo `MATERIAS_ACTIVAS`.
- Familias de delito: 25 (nueva «cuestiones procesales» para quejas, excusas y conflictos de competencia; plagio,
  prostitución infantil, uso de documento falso, sustracción de menores y ejercicio indebido del servicio público con su
  familia). `familias_penal.csv` lista los 546 tipos; ninguna asignación revisada por un especialista.
- Juez `gemma2:9b` con el corpus final: 0.742 en 725 sentencias con las 25 familias (`20261121-031724_723fe6b`), **no se usa** (mínimo 0.80 fijado de antemano;
  `20261121-003449_5311dc1_referencias`). La referencia de acto, apelante y sentido queda con las reglas.

### Quitado
- Versiones pasadas de los entregables en `docs/` (informe v8 y presentación anterior), reemplazadas por la presentación
  v11.2; siguen en el historial de git.
- Código sin uso: `f1_conjuntos`, `coincide_texto` y el atributo `veredicto_relevancia`.
- Corridas obsoletas que nada citaba (una regresión intermedia y dos coberturas de metadatos de corpus intermedios).
- Camino de anotación humana, que nunca se usó: hojas de juicios y fichas, piloto de anotación (`hojas-anotacion`,
  `kappa-piloto`), `generacion`, estudio de uso (`estudio`), consultas reales vacías, plantilla inicial (`plantilla`)
  y la opción `COPILOTO_EVALUACION_AUTOMATICA`. Las hojas estaban vacías.
- `evaluacion modelos-tareas`: no se corrió y no respalda ninguna decisión.
- Listas de descarga viejas (`redescarga_penal.csv`, `pendientes_penal.csv`), reemplazadas por `evaluacion/descargas/`.

## [11.1.0] - 2026-11-13

Preparación de la evaluación final: lo evaluado queda congelado, la partición de prueba se usa una sola vez y las
fichas que ve la app son las mismas que anota el abogado. Ningún número nuevo sale de fuera de
`evaluacion/resultados/`.

### Añadido
- `evaluacion congelar` y control de la partición de prueba: huella de código, corpus, índice, fichas, configuración
  efectiva, texto de cada prompt y digest del modelo; las corridas de prueba se niegan a correr si algo cambió y solo
  corren una vez por comando (`--forzar "motivo"`, que queda en el reporte). Todos los reportes llevan el digest del
  modelo y las semillas.
- Fichas penales precalculadas (`evaluacion fichas-batch`, reanudable): la app usa la ficha guardada si coinciden el
  texto y la versión; la hoja de fichas muestra esa misma ficha al abogado. Diagnóstico sin anotación de las 1,412
  fichas (JSON inválido, sentido LLM vs. reglas, artículos no encontrados, "dato no disponible" por campo).
- Metadatos procesales por reglas para todo el corpus penal (acto impugnado, quién apeló, sentido del fallo), con su
  cobertura, y orden de candidatos por facetas: si la consulta nombra el acto impugnado, primero los de la misma
  familia y el mismo acto (en prisión preventiva por homicidio, los 3 primeros pasan de robo, robo, homicidio a
  homicidio x3).
- Conjunto de evaluación: regeneración estricta de los borradores marcados, `consultas_reales.jsonl` (consultas de
  abogados sin ver sentencias, reportadas aparte), piloto de anotación con `kappa-piloto`, columnas de faceta
  (mismo delito / mismo problema procesal) y efecto mínimo detectable en cada comparación pareada.
- Re-ranking con cross-encoder (`BAAI/bge-reranker-v2-m3`) como variante de `recuperacion`; apagado por defecto.
- Síntesis verificable: cada afirmación cita su precedente ([P1], [P2]...) y el código marca "verificar" si la cita no
  existe o el sentido del fallo o el delito no coinciden con la ficha citada.
- `copiloto-datos redescarga` (`evaluacion/redescarga_penal.csv`, ordenada por familia con menos sentencias); al
  reintegrar los PDF, la versión completa reemplaza a la muestreada.
- GitHub Actions: pyflakes y pytest en cada push, sin GPU ni Ollama, con una prueba de punta a punta (corpus mínimo,
  índice, Chroma, BM25, clasificador y pipeline) con embeddings y LLM simulados.
- `evaluacion errores --corrida ID`: un caso por fallo, clasificado en el primer paso que falla (cobertura,
  clasificación, orden, descarte de QC_1, ficha, otro), con su conteo y `evaluacion/errores/casos_presentacion.md`.
- `evaluacion modelos`: compara los modelos de Ollama ya instalados (lista en `MODELOS_COMPARACION`; los que faltan se
  omiten, no se descargan) con la misma semilla, temperatura 0 y las mismas consultas y sentencias: latencia, llamadas,
  fichas sin anotación y VRAM máxima.
- `evaluacion robustez`: consultas con faltas de ortografía, muy cortas y muy largas, de otra materia o tema, con datos
  personales inventados (que no deben quedar en ningún archivo) y con instrucciones inyectadas.
- Resultados guardados para la demo (`CACHE_RESULTADOS`, `cli demo --ejemplos`): la app muestra "Resultado guardado
  el <fecha>" si la consulta ya se analizó con la misma configuración, modelo, índice, fichas y código.
- Estudio de uso manual vs. app (`evaluacion/estudio_uso/`, `evaluacion estudio`): 5 casos en borrador, registro con
  asignación balanceada e instrucciones; con menos de 10 participantes se reporta como exploratorio.
- `fichas-batch` reintenta si el servidor de Ollama se cae y guarda el avance antes de rendirse.

### Corridas (diagnóstico, commit `fb1f3fe`)
- Fichas: `20261113-105003_fb1f3fe_fichas-diagnostico`. Latencia sin y con fichas precalculadas:
  `20261113-111047_fb1f3fe`, `20261113-111602_fb1f3fe`, `20261113-112440_fb1f3fe`. QC_1:
  `20261113-114412_fb1f3fe_qc-diagnostico`. Regresión: `20261113-114442_fb1f3fe`. Robustez:
  `20261113-115133_fb1f3fe`. Demo: `20261113-120016_fb1f3fe`. Modelos: `20261113-125326_fb1f3fe`. Recuperación en
  desarrollo (consultas en borrador, no reportable): `20261113-130208_fb1f3fe`.

### Cambiado
- Limpieza del texto: restos del encabezado del tribunal intercalados con el expediente, entradas del anexo cortadas
  por páginas omitidas y encabezados repetidos al final; corpus e índice regenerados.
- `generacion` y QC_2 en `qc` juzgan la ficha y el control que usa la app (antes, los de v10).
- La búsqueda de candidatos (clasificación + búsqueda adaptativa) es una sola función para la app y la evaluación.

## [11.0.0] - 2026-11-12

Respuesta a la evaluación del jurado: alcance acotado a la defensa penal en apelación, evaluación reproducible y
explicación de cada número. Ningún resultado se reporta si no está en una corrida guardada en `evaluacion/resultados/`.

### Añadido
- Módulo `evaluacion`: métricas (Precision/Recall/Hit@k, MRR, nDCG, F1 macro, kappa, IC de Wilson y bootstrap
  pareado) probadas con los ejemplos resueltos del informe; familias de delito (289 tipos penales -> 24 familias,
  CSV para revisión de un abogado); conjuntos de consultas, juicios y fichas con validación y detección de fuga;
  experimentos `recuperacion`, `ablacion`, `pesos`, `qc`, `qc-diagnostico`, `generacion`, `latencia`, `regresion`;
  borradores de consultas con el LLM (`revisado: false`, la partición de prueba se rechaza hasta revisarlos) y hojas
  de anotación CSV.
- Ficha penal orientada a la defensa: acto impugnado, apelante, agravios y su calificación, sentido del fallo con
  valores cerrados; generada en una sola llamada (JSON validado con reintento).
- Verificaciones deterministas de la ficha: sentido del fallo leído de los resolutivos, artículos citados contra el
  texto de la sentencia y QC_2 por familia del delito; las discrepancias se marcan "verificar". Si los
  resolutivos contradicen a la ficha, la ficha completa y el informe usan el sentido de los resolutivos.
- Síntesis v11: la lista de precedentes favorables o desfavorables para la defensa se calcula con reglas (quién
  apeló + sentido del fallo; también en cada tarjeta); el LLM resume los argumentos aceptados o rechazados y no
  pronostica el resultado (la v10 leyó al revés un precedente en una consulta real).
- Limpieza del texto: anexo de datos testados, etiquetas de testado (también en femenino, plural o partidas),
  encabezados y pies de página aunque la extracción parta las palabras, números de expediente repetidos en cada
  hoja y palabras deletreadas; idempotente
  (`cli limpiar-corpus`, 16 % menos caracteres en el corpus guardado). Las sentencias penales se indexan completas.
  Comandos `cli pendientes` y `estado --salida`; lista de sentencias con páginas omitidas en
  `evaluacion/paginas_omitidas.csv`.
- Página "Cómo funciona" con el flujo, los límites de tamaño, glosario y la última evaluación guardada.

### Cambiado
- Por defecto: solo materia penal, clasificación por familia de delito, prompts v11; las opciones de v10 quedan por
  variable de entorno para comparar.
- Clasificador: Jaccard 0.40 y KEM 0 (eran redundantes); sin la lista fija de tokens penalizados; el embedding de
  cada familia incluye los nombres de sus tipos, sin el sufijo "y otro(s)" del portal ni nombres compuestos con
  otro delito; con una sola materia, "penal" ya no cuenta para Jaccard. Decisiones respaldadas por la regresión
  guardada.
- QC_1: lee el fragmento más parecido a la consulta y acepta mismo delito, misma familia o mismo problema
  procesal; si descarta todo, se muestran los mejores candidatos marcados "no verificado".
- Evaluación del conjunto A con *leave-one-out*: la sentencia fuente no vota por su propio tipo.
- Índice: sentencias penales completas (antes 12 fragmentos); PDF leídos completos (antes se muestreaban los de más
  de 50 páginas); recorte para el LLM que conserva inicio, análisis de agravios y resolutivos.
- Interfaz: pantalla principal con consulta, pasos, síntesis y tarjetas; detalles técnicos en un solo desplegable;
  configuración en la barra lateral, cerrada. Ayudas de una frase en cada número visible.
- LLM con semilla fija (42); embeddings en CPU si la GPU no tiene memoria.

### Eliminado
- Páginas "Datos" (sus herramientas están en la CLI) y "Evaluación" (integrada en "Cómo funciona").
- Ejemplos de otras materias en la app (se conservan como regresión en `evaluacion/regresion.py`).
- KEM sobre vocabulario y su archivo de términos (perdió en la regresión).

## [10.0.0] - 2026-10-30

### Añadido
- Interfaz web en Streamlit: consultas de ejemplo etiquetadas y editables, progreso animado por paso, logs en vivo,
  síntesis en streaming, resultados en pestañas y exportación a Markdown.
- Página **Datos** con el balance del corpus y herramientas de descarga, procesamiento y reindexación.
- Selector de LLM: Ollama o cualquier API compatible con OpenAI, configurable desde la interfaz o con `.env`.
- Búsqueda híbrida embeddings + BM25 con *Reciprocal Rank Fusion*.
- Voto de recuperación en la clasificación híbrida y filtro estricto por las 3 mejores categorías.
- Descargador balanceado (tope de 30 por tipo, validación de PDFs completos, detección de reCAPTCHA).
- Corpus balanceado (3,544 sentencias, 567 tipos) e índice de embeddings versionados en el repositorio.
- Paquete `copiloto_juridico`, CLI de datos, pruebas unitarias, `pyproject.toml`, `.env.example`.

### Cambiado
- Embeddings: `all-mpnet-base-v2` → `intfloat/multilingual-e5-base`, con *mean pooling* enmascarado,
  normalización L2 y fragmentación de cada sentencia.
- Las respuestas del LLM conservan mayúsculas y formato Markdown.
- Se fija `num_ctx` en Ollama y se recorta el contexto conservando el inicio y el final de cada sentencia.
- Las categorías del clasificador se derivan del corpus indexado.
- Notebooks movidos a `notebooks/` y documentos a `docs/`.

### Corregido
- QC_1 (relevancia) se había eliminado y aceptaba cualquier respuesta que contuviera "si"; QC_2 nunca se ejecutaba.
- `generate_final_synthesis` usaba marcadores que no existían en el prompt.

### Eliminado
- Cachés `rag_categories_cache7.json` / `rag_embeddings_cache7.npy` (no correspondían a la base en uso).

## [9.8] - 2025-11

- Versión en notebook (`notebooks/proyecto_8_llm_sentencia_judicial.ipynb`): pipeline RAG con clasificación híbrida,
  búsqueda adaptativa, extracción por materia y síntesis final.
