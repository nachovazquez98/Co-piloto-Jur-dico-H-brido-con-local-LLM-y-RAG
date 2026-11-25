<div align="center">

# ⚖️ Co-piloto Jurídico

**Búsqueda y análisis de precedentes penales de apelación del Supremo Tribunal de Justicia de Jalisco, con búsqueda
híbrida (embeddings + BM25), un LLM local y una interfaz web.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App_web-ff4b4b?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-LLM_local-blueviolet)
![Version](https://img.shields.io/badge/versión-11.2.0-informational)
![License](https://img.shields.io/badge/licencia-MIT-green)

</div>

---

## Objeto y alcance

El portal del Supremo Tribunal de Justicia del Estado de Jalisco (STJ) localiza las sentencias por categoría del
delito, toca o fecha; conocer los hechos, los agravios y cómo los calificó la Sala exige leer cada resolución. A
partir de la descripción de un caso, el sistema recupera las resoluciones penales más afines, elabora una ficha de
cada una (delito, acto impugnado, agravios y su calificación, sentido del fallo, fundamento legal) y una síntesis
comparativa. Todo se ejecuta en el propio equipo: la descripción del caso no sale de él.

| Perfil | Uso del sistema |
|---|---|
| **Abogado** (defensor o asesor jurídico de la víctima) — usuario principal | Precedentes en que la Sala calificó agravios semejantes como fundados, infundados o inoperantes |
| **Tribunal de alzada** (Sala Penal) | Contraste del criterio de las Salas en supuestos semejantes; el sentido de la resolución corresponde exclusivamente a la Sala |

La consulta de una persona sin formación jurídica queda fuera del alcance: no tiene perfil ni se ha evaluado. El
sistema no pronostica el resultado, no redacta agravios y no sustituye la lectura de la resolución original. Las
resoluciones del STJ son precedentes orientadores, no jurisprudencia obligatoria.

## Instalación

Requisitos: Python 3.10+ y [Ollama](https://ollama.com) con `llama3.1:8b` (4.9 GB). Una GPU NVIDIA de 8 GB acelera
todo, pero no es obligatoria.

```bash
git clone https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG.git
cd Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
ollama pull llama3.1:8b
streamlit run app/Copiloto.py
```

El repositorio trae todo lo que la app necesita: el texto de las 2,495 sentencias (`data/corpus/`), sus embeddings
(`data/indice/`) y las fichas precalculadas (`data/fichas/`). En el primer arranque se descarga el modelo de
embeddings (~1 GB) y se construye la base local (unos minutos); después la app abre en unos 8 s.

**PDF originales (opcional).** No están en git: solo se necesitan para reconstruir el corpus. Se descargan de la
versión publicada [`v11.2`](https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG/releases/tag/v11.2)
(15 ZIP, 1.2 GB):

```bash
for i in $(seq -w 1 15); do
  curl -LO https://github.com/nachovazquez98/Co-piloto-Jur-dico-H-brido-con-local-LLM-y-RAG/releases/download/v11.2/sentencias-penales-pdf-$i-de-15.zip
done
for f in sentencias-penales-pdf-*-de-15.zip; do unzip -q -o "$f" -d data/raw/; done
```

Para leer PDF escaneados al construir el corpus hacen falta `tesseract` (con el modelo de español `spa`) y
`pdftoppm` (poppler); sin ellos, esas páginas se omiten.

## Uso

1. Elige el **perfil** y describe el caso (hechos, delito, resolución apelada y agravios), o carga un ejemplo.
2. Pulsa **Buscar precedentes**.
3. Lee la síntesis y las tarjetas. Donde aparezca **verificar** o **sin verificar**, revisa la sentencia original
   antes de citarla. El informe se descarga en Markdown.

La página **Cómo funciona** explica el flujo, el glosario jurídico y la última evaluación guardada.

## Cómo funciona

| Paso | Entrada → salida | Técnica |
|---|---|---|
| 1. Conceptos clave | consulta → delito + problema procesal | LLM local (`llama3.1:8b`, Ollama) |
| 2. Clasificación | consulta → 3 familias de delito más probables | coseno de embeddings + Jaccard de lemas + voto de las sentencias recuperadas |
| 3. Búsqueda | familias o delitos elegidos → hasta 30 candidatas | `multilingual-e5-base` (ChromaDB) + BM25, fusión RRF |
| 4. Fichas | candidata → ficha penal | orden por familia y acto impugnado; ficha en una llamada (JSON validado); sentido del fallo por reglas, artículos contra el texto, QC_2 por familia |
| 5. Síntesis | fichas → respuesta | parte favorecida por reglas; el LLM resume, solo con las fichas, los argumentos que la Sala aceptó o rechazó, sin pronosticar |

**Datos:** 2,495 sentencias penales públicas del STJ (máximo 30 por tipo de delito); sus 546 tipos del portal se
agrupan en 25 familias ([`evaluacion/familias_penal.csv`](evaluacion/familias_penal.csv)).

## Resultados

Partición de prueba, una sola corrida por comando (congelado `a15b83c852281010`, commit `4b1529e`), 46 consultas,
IC 95 %. No hubo abogados disponibles: la referencia es automática (categoría del STJ, reglas sobre el texto y un juez
LLM) y se describe en [`evaluacion/README.md`](evaluacion/README.md).

| Qué se mide | Resultado | Corrida |
|---|---|---|
| Familia correcta entre las 3 del filtro (Hit@3) | **0.717** [0.575, 0.827]; línea base k-NN 0.435 (+0.283, significativa) | `20261121-032545_4b1529e_recuperacion` |
| Búsqueda híbrida (nDCG@5) | 0.285; BM25 0.243; embeddings 0.304; diferencias no significativas | `20261121-032545_4b1529e_recuperacion` |
| Exactitud de la ficha | familia **0.917**, acto impugnado 0.883, apelante 0.830, sentido del fallo 0.722 | `20261121-032553_4b1529e_fichas-referencia` |
| Fundamento legal de la ficha | precisión **0.644**; exhaustividad 0.133 (la ficha resume los principales) | `20261121-032553_4b1529e_fichas-referencia` |
| Ahorro de trabajo (sentencias que hay que abrir para encontrar 3 útiles) | mediana 22 frente a 31 en el portal | `20261121-032625_4b1529e_esfuerzo` |
| Robustez (8 consultas difíciles) | 8/8 terminan; ninguna síntesis pronostica; ningún archivo guarda datos personales | `20261121-053544_4b1529e_robustez` |

**Después de la evaluación final**, en la partición de desarrollo y por pares (detalle en
[`CHANGELOG.md`](CHANGELOG.md)):

- **Precedentes que ve el usuario:** con el orden por familia y acto impugnado, P@3 **0.410** frente a 0.248 sin ese
  orden (+0.162 [+0.077, +0.248]) (`20261124-014247_0f2894a_precedentes`).
- **QC_1 apagado por defecto:** no cambia los precedentes mostrados (P@3 0.410 con y sin él) y la consulta pasa de
  30.8 s a **17.1 s** (p50) y de 5 a 2 llamadas al modelo (`20261124-020004_2e50981_latencia`). Sigue disponible en
  Configuración.
- **Descartados**, por no mejorar o empeorar otra medida: QC_1 con `qwen2.5:7b` o sin razonar, dos variantes de la
  ficha para el fundamento legal y la síntesis con ejemplos.

| Latencia por consulta (10 ejemplos, RTX 4060 8 GB) | Total p50 | Total p95 | Llamadas al LLM | Corrida |
|---|---|---|---|---|
| Ficha en dos llamadas + QC_2 con el LLM | 321 s | 847 s | 17 | `20261112-175942_c78b576` |
| Ficha en una llamada + QC_2 por reglas | 138 s | 168 s | 8 | `20261112-173610_0a20152` |
| Además, fichas precalculadas | 31 s | 53 s | 5 | `20261121-052935_4b1529e` |
| Además, QC_1 apagado (la app) | **17 s** | **35 s** | 2 | `20261124-020004_2e50981` |

## Limitaciones

- **Cobertura:** muchos tipos penales tienen una o ninguna sentencia; la app avisa cuando no hay precedentes de las
  familias detectadas o la consulta no parece penal.
- **Consultas que no nombran el acto impugnado:** la precisión de los precedentes cae (P@3 0.246 frente a 0.567 en
  desarrollo); la app pide nombrarlo.
- **Sentido del fallo:** la ficha acierta en 0.722; la tarjeta usa el sentido leído de los puntos resolutivos
  (1,966 de 2,495 sentencias) y marca **sin verificar** el resto.
- **Perfil del tribunal:** con consultas reescritas como toca en estudio, la familia se encuentra igual, pero el orden
  de los precedentes empeora (nDCG@5 0.266 → 0.145; `20261123-202205_e566e3d-sucio_perfil-tribunal`).
- **Evaluación sin especialistas:** mide si el sistema encuentra precedentes del mismo delito y problema procesal,
  no si un abogado los consideraría útiles.
- **Ventana del LLM:** de una sentencia larga se envían el inicio, el análisis de los agravios y los resolutivos.
- **Datos testados:** nunca se infieren. Un PDF escaneado del proyecto no se incorporó porque no tiene suprimidos los
  datos personales.

## Próximos pasos

1. **Fundamento legal:** el error es de selección, no de invención (126 de 127 artículos fuera del fundamento sí
   están en la sentencia); requiere un modelo mayor o validación con especialistas.
2. **Tribunal de alzada:** evaluar con consultas redactadas por personal de una Sala e integrar la búsqueda en el
   portal del STJ.
3. **Consultas sin acto impugnado:** deducir el acto de la descripción y medirlo en una partición nueva.
4. **Corpus:** ampliarlo con [`descarga_datos/`](descarga_datos/) (una persona resuelve el reCAPTCHA) y volver a
   descargar las sentencias con páginas omitidas.
5. **Validación con especialistas** de la referencia automática.

## Línea de comandos

```bash
python -m copiloto_juridico.cli estado                              # balance del corpus
python -m copiloto_juridico.cli construir-corpus --pdfs carpeta/    # añadir PDF
python -m copiloto_juridico.cli indexar                             # regenerar embeddings (GPU recomendada)
python -m copiloto_juridico.cli demo --ejemplos                     # dejar guardados los ejemplos
python -m copiloto_juridico.evaluacion --help                       # evaluación (ver evaluacion/README.md)
```

**Configuración:** los valores por defecto están en `config.py`, donde cada decisión cita la corrida que la respalda;
se cambian con variables de entorno (`.env.example`) o desde la app. La app guarda cada resultado en
`data/cache/resultados/` (local, fuera de git); si las consultas traen datos de clientes, apágalo con
`COPILOTO_CACHE_RESULTADOS=false`.

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

Las pruebas no requieren GPU ni Ollama. El notebook técnico está en [`notebooks/`](notebooks/) y la presentación con
sus notas del orador en [`docs/`](docs/). Historial en [`CHANGELOG.md`](CHANGELOG.md); guía para contribuir en
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licencia y créditos

Código bajo licencia **MIT** ([`LICENSE`](LICENSE)). Las sentencias son documentos públicos del Supremo Tribunal de
Justicia del Estado de Jalisco. Desarrollo: **Ignacio Vázquez**
([@nachovazquez98](https://github.com/nachovazquez98)). Modelos: Meta Llama 3.1, multilingual-e5-base, spaCy.

## Aviso legal

Herramienta académica de apoyo basada en sentencias públicas. **No constituye asesoría jurídica.** Las fichas las
genera un modelo de lenguaje y pueden contener errores u omisiones: verifica siempre la sentencia original.
