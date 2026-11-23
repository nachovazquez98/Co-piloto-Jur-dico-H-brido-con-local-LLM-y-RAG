"""Prompts del Co-piloto Jurídico (migrados del notebook v9.8).

Cambios respecto al notebook:
- Los verificadores (QC_1 y QC_2) terminan con una línea `VEREDICTO: SI|NO` que se parsea de forma estricta.
- Los informes piden Markdown explícitamente, porque la app los muestra tal cual.
"""

PROMPT_EXTRACCION_ENTIDADES = """
Tu única tarea es analizar la Consulta del Usuario y extraer de 1 a 3 conceptos jurídicos clave que sirvan como etiquetas de búsqueda.
Es FUNDAMENTAL que respondas ÚNICAMENTE con los conceptos separados por comas, sin ningún tipo de explicación, preámbulo, disculpa o texto adicional. Si no puedes extraer, responde solo: 'FALLA'.
Consulta del Usuario: "{texto_consulta}"
Conceptos Clave:
"""

# v11: una consulta de defensa suele centrarse en el problema procesal ("vinculación a proceso", "reconocimiento
# ilegal") y el LLM extrae solo esos conceptos; entonces Jaccard y KEM valen 0 para todas las categorías (que son
# delitos). Esta variante pide primero el delito. Se activa con COPILOTO_PROMPT_CONCEPTOS=v11.
PROMPT_EXTRACCION_ENTIDADES_V11 = """
Tu única tarea es analizar la Consulta del Usuario y extraer de 1 a 3 conceptos jurídicos clave que sirvan como etiquetas de búsqueda.
El PRIMER concepto debe ser SIEMPRE el nombre del delito, con su modalidad si se menciona o se deduce (ej: 'robo calificado', 'homicidio a título de culpa', 'violencia familiar'). Si la consulta no nombra el delito, dedúcelo de los hechos descritos (ej: "le quitaron la cartera amenazándolo con una navaja" -> 'robo calificado'). Los demás, si los hay, son el problema procesal (ej: 'vinculación a proceso', 'prisión preventiva', 'detención ilegal').
Es FUNDAMENTAL que respondas ÚNICAMENTE con los conceptos separados por comas, sin ningún tipo de explicación, preámbulo, disculpa o texto adicional. Si no puedes extraer, responde solo: 'FALLA'.
Consulta del Usuario: "{texto_consulta}"
Conceptos Clave:
"""


def prompt_conceptos(version: str | None = None) -> str:
    """Prompt del paso 1 (conceptos clave) según `config.PROMPT_CONCEPTOS_VERSION`."""
    from . import config

    version = version or config.PROMPT_CONCEPTOS_VERSION
    if version not in ("v10", "v11"):
        raise ValueError(f"Versión de prompt de conceptos desconocida: {version}")
    return PROMPT_EXTRACCION_ENTIDADES if version == "v10" else PROMPT_EXTRACCION_ENTIDADES_V11


PROMPT_VERIFICACION_RELEVANCIA = """
Eres un Asistente de Verificación de Relevancia legal. Tu única tarea es ser estricto y literal.

Compara el **Contexto Específico** de la Consulta del Usuario con el **Contexto Específico** del Documento Recuperado.

Responde SI ÚNICAMENTE si el documento trata sobre el MISMO tema jurídico Y un contexto compatible.
Responde NO si los contextos son diferentes, aunque compartan palabras clave.
**EJEMPLO IMPORTANTE:** Si la consulta es sobre "lesiones en una riña de bar" (contexto: pelea pública) y el documento
es sobre "lesiones por violencia familiar" (contexto: doméstico), DEBES RESPONDER NO.

**Consulta del Usuario:**
"{texto_consulta}"

**Fragmento del Documento Recuperado:**
"{fragmento_documento}"

Responde exactamente con este formato:
1. Contexto de la consulta: <una frase>
2. Contexto del documento: <una frase>
VEREDICTO: SI o NO
"""

# v11: el criterio es el que usa un defensor para decidir si una sentencia le sirve como precedente.
PROMPT_VERIFICACION_RELEVANCIA_V11 = """
Eres un Asistente de Verificación de Relevancia para un defensor penal. Decide si el Documento Recuperado puede
servirle como precedente para la Consulta.

Responde SI si el documento trata:
- el MISMO delito o un delito de la MISMA familia (ej: robo calificado y robo equiparado; homicidio calificado y
  homicidio en grado de tentativa), o
- el MISMO problema procesal (ej: prisión preventiva, vinculación a proceso, legalidad de la detención,
  valoración de un reconocimiento), aunque el delito sea otro.
No hace falta que los hechos sean idénticos.

Responde NO si el documento trata otro delito Y otro problema procesal, aunque compartan palabras.
**EJEMPLO NO:** consulta sobre "lesiones en una riña de bar" (lesiones entre desconocidos en un lugar público) y
documento sobre "lesiones por violencia familiar" (otro delito: protege la integridad familiar, contexto doméstico).

**Consulta del Usuario:**
"{texto_consulta}"

**Fragmento del Documento Recuperado:**
"{fragmento_documento}"

Responde exactamente con este formato:
1. Delito y problema de la consulta: <una frase>
2. Delito y problema del documento: <una frase>
VEREDICTO: SI o NO
"""


def prompt_qc1(version: str | None = None) -> str:
    """Prompt de QC_1 según `config.PROMPT_QC1_VERSION` ("v10": mismo tema Y contexto; "v11": criterio de defensa)."""
    from . import config

    version = version or config.PROMPT_QC1_VERSION
    if version not in ("v10", "v11"):
        raise ValueError(f"Versión de QC_1 desconocida: {version}")
    return PROMPT_VERIFICACION_RELEVANCIA if version == "v10" else PROMPT_VERIFICACION_RELEVANCIA_V11


PROMPT_VERIFICACION_POST_REPORTE = """
Tu única tarea es actuar como un verificador de CALIDAD.
Compara la Consulta Original del Usuario con el Informe de Análisis que se acaba de generar.

Responde NO si el informe:
a) Habla de un tema legal completamente diferente (ej. consulta sobre "lesiones" y el informe habla de "fraude").
b) Contiene alucinaciones o información sin relación (ej. menciona "falta de pago" en un caso penal de lesiones).
En cualquier otro caso, si el informe es útil como precedente para la consulta, responde SI.

**Consulta Original del Usuario:**
"{texto_consulta}"

---
**Informe de Análisis Generado:**
"{informe_generado}"
---

Responde exactamente con este formato:
1. Relevancia: <una frase>
2. Alucinaciones: <una frase>
VEREDICTO: SI o NO
"""

_NOTA_ANONIMIZACION = """**NOTA IMPORTANTE: El texto puede contener datos anonimizados como '[ELIMINADO]', 'eliminado' o '[No.1]...'.
IGNORA esas marcas. Si un dato no se puede encontrar (ya sea porque fue eliminado o no se menciona),
DEBES escribir [DATO NO DISPONIBLE] en su lugar.** Responde en español."""

PROMPT_EXTRACCION_PENAL = """
Eres un Analista Legal Penal. Tu misión es extraer la información más exhaustiva y crítica de la SENTENCIA PENAL.

""" + _NOTA_ANONIMIZACION + """

**Formato de Extracción (Detallado):**
- **Tribunal:** [Nombre completo de la Sala/Juzgado que resolvió, incluyendo la fecha exacta de la resolución]
- **Identificador Judicial:** [Toca/Expediente/Causa, tal como aparece]
- **Etapa Procesal:** [Ej: Sentencia Definitiva de Segunda Instancia, Auto de Vinculación a Proceso]
- **Delito(s) Principal(es):** [Lista completa y precisa de los delitos imputados/acreditados (ej: Robo Calificado Art. 233)]
- **Acusado(s):** [Nombre(s) o identificador(es) del acusado/sentenciado.]
- **Víctima(s)/Ofendido(s):** [Nombre(s) o identificador(es) de la víctima/ofendido.]
- **Hechos Clave:** [Narrativa breve y fáctica de la acción delictiva, las circunstancias de modo, tiempo y lugar.]
- **Decisión Específica:** [Fallo exacto: Se MODIFICA/CONFIRMA/REVOCA la sentencia, o la pena final impuesta (años/multa).]
- **Fundamento Legal:** [Enumera los artículos, códigos y jurisprudencias exactas citados como base de la decisión.]
- **Pruebas Decisivas:** [Enumera las pruebas cruciales que sustentaron el fallo (ej: Testimonial del ofendido, Peritaje balístico).]

Contexto (Sentencia):
{contexto}

Hechos Extraídos:
"""

_INSTRUCCION_FORMATO = """Responde ÚNICAMENTE con el informe en Markdown, en español, con redacción profesional
(mayúsculas y acentos correctos), sin preámbulos ni comentarios finales. Copia los datos tal como aparecen en los
Hechos Extraídos; si un dato dice [DATO NO DISPONIBLE], déjalo así."""

PROMPT_REPORTE_PENAL = """
Con base en los siguientes HECHOS EXTRAÍDOS de una sentencia penal, genera un informe legal profesional, **detallado y analítico**.
Utiliza los 'Hechos Extraídos' para construir una narrativa coherente y explicar las razones del tribunal.
""" + _INSTRUCCION_FORMATO + """

**Hechos Extraídos:**
{hechos}

**Formato del Informe:**
#### 1. Resumen Ejecutivo
[Resumen breve (2-3 frases) del delito principal y la decisión final del tribunal sobre el acusado.]

#### 2. Datos de Identificación del Proceso
- **Tribunal:** [dato 'Tribunal']
- **Expediente:** [dato 'Identificador Judicial']
- **Etapa Procesal:** [dato 'Etapa Procesal']
- **Delito(s):** [dato 'Delito(s) Principal(es)']

#### 3. Partes Involucradas
- **Acusado(s):** [dato 'Acusado(s)']
- **Víctima(s)/Ofendido(s):** [dato 'Víctima(s)/Ofendido(s)']

#### 4. Análisis de Hechos y Decisión
**Hechos Clave:** [narrativa de 'Hechos Clave']

**Decisión del Tribunal:** [dato 'Decisión Específica']

**Razonamiento del Tribunal:** [Con base en 'Fundamento Legal' y 'Pruebas Decisivas', **explica** por qué el tribunal tomó esa decisión. ¿Qué pruebas fueron más importantes? ¿Cómo se aplicaron los artículos citados?]
"""

PROMPT_SINTESIS_FINAL = """
Eres un Abogado Coordinador experto. Tu trabajo es revisar los informes de análisis de precedentes que prepararon tus asistentes y sintetizarlos en un solo informe final.

**BASÁNDOTE ÚNICA Y EXCLUSIVAMENTE en los informes proporcionados**, responde la Consulta Original del Usuario. NO inventes información ni uses conocimiento externo.
Cuando cites un precedente, refiérete a él como "Precedente 1", "Precedente 2", etc.
""" + _INSTRUCCION_FORMATO + """

Consulta Original del Usuario:
{text_consulta}

---
INFORMES DE PRECEDENTES (PARA ANALIZAR):
{informes_precedentes}
---

Formato de Salida Requerido (Markdown):

### 1. Resumen Ejecutivo
[Un párrafo que resuma la conclusión general de los informes, respondiendo directamente a la consulta del usuario SÓLO con datos de los informes.]

### 2. Precedente Más Relevante
[Identifica cuál de los precedentes es el más relevante y explica por qué.]

### 3. Análisis Comparativo
- **Puntos en Común:** [Argumentos o fallos que se repiten en los informes.]
- **Diferencias y Matices:** [Contradicciones o detalles únicos de cada informe.]

### 4. Conclusión para el Usuario
[Conclusión final de 1-2 frases basada SÓLO en la evidencia combinada de los informes.]
"""

PROMPT_SINTESIS_FINAL_V11 = """
Eres un abogado penalista que analiza precedentes de apelación. Abajo están las fichas de precedentes de las Salas
Penales del STJ Jalisco. Responde la consulta (de un abogado defensor o asesor jurídico de la víctima, o del tribunal de
alzada) usando ÚNICA Y EXCLUSIVAMENTE esas fichas.

Reglas:
- Termina CADA frase que afirme algo de un precedente con su número entre corchetes: [P1], [P2] (o [P1, P3] si son
  varios). Una frase sin precedente que la respalde no se escribe.
- Atribuye a cada precedente solo lo que dice SU ficha. Si la ficha no menciona algo (por ejemplo, un tipo de prueba),
  no digas que el precedente lo analiza.
- Quién apeló y qué parte resultó favorecida ya están calculados en cada ficha ("Apelante", "Parte favorecida"); el
  sistema los muestra aparte. No los repitas ni los contradigas.
- NO predigas el resultado del caso consultado ni digas que una apelación "tendrá éxito". Di qué argumentos acogió
  o rechazó la Sala y por qué.
""" + _INSTRUCCION_FORMATO + """

Consulta:
{text_consulta}

---
FICHAS DE PRECEDENTES:
{informes_precedentes}
---

Formato de salida (Markdown):

### Respuesta breve
[2 o 3 frases: qué muestran estos precedentes sobre el problema de la consulta, cada una con su cita [Pn].]

### Argumentos que la Sala aceptó o rechazó
[Una viñeta por precedente: qué argumento de los agravios aceptó o rechazó la Sala y por qué, según su ficha. [Pn]]

### Qué revisar en las sentencias
[1 o 2 frases: qué partes de qué precedentes conviene leer completas, con su cita [Pn]. Sin pronosticar.]
"""


def prompt_sintesis(version: str | None = None) -> str:
    """Prompt de la síntesis según `config.PROMPT_SINTESIS_VERSION` ("v10" o "v11": a favor / en contra, sin
    pronosticar)."""
    from . import config

    version = version or config.PROMPT_SINTESIS_VERSION
    if version not in ("v10", "v11"):
        raise ValueError(f"Versión de síntesis desconocida: {version}")
    return PROMPT_SINTESIS_FINAL if version == "v10" else PROMPT_SINTESIS_FINAL_V11




# --- v11: ficha penal orientada a la defensa en apelación (config.PROMPT_PENAL_VERSION = "v11") ---
PROMPT_EXTRACCION_PENAL_V11 = """
Eres un Analista Legal Penal que apoya a la DEFENSA en segunda instancia. Extrae de la SENTENCIA PENAL (resolución de
un recurso de apelación de una Sala Penal) la información que un defensor necesita para usarla como precedente.

""" + _NOTA_ANONIMIZACION + """

**Formato de Extracción (Detallado):**
- **Tribunal:** [Nombre completo de la Sala que resolvió, incluyendo la fecha exacta de la resolución]
- **Identificador Judicial:** [Toca/Expediente/Causa, tal como aparece]
- **Etapa Procesal:** [Etapa del proceso en la que se dictó la resolución apelada (ej: etapa de investigación complementaria, etapa intermedia, juicio oral)]
- **Acto Impugnado:** [Qué resolución se apeló: auto de vinculación a proceso, auto de no vinculación, medida cautelar (ej. prisión preventiva), sentencia definitiva de juicio oral, sobreseimiento, u otro]
- **Apelante:** [Quién interpuso el recurso: defensa, Ministerio Público, asesor jurídico de la víctima]
- **Delito(s) Principal(es):** [Lista completa y precisa de los delitos imputados/acreditados (ej: Robo Calificado Art. 233)]
- **Acusado(s):** [Nombre(s) o identificador(es) del acusado/sentenciado.]
- **Víctima(s)/Ofendido(s):** [Nombre(s) o identificador(es) de la víctima/ofendido.]
- **Hechos Clave:** [Narrativa breve y fáctica de la acción delictiva, las circunstancias de modo, tiempo y lugar.]
- **Agravios Planteados:** [Lista numerada y breve de los argumentos del apelante contra la resolución]
- **Calificación de los Agravios:** [Para cada agravio, cómo lo calificó la Sala: FUNDADO, INFUNDADO o INOPERANTE, con una frase de por qué]
- **Sentido del Fallo:** [EXACTAMENTE una de estas palabras: CONFIRMA, MODIFICA, REVOCA, REPONE u OTRO]
- **Decisión Específica:** [Detalle del fallo: qué se confirma, modifica o revoca; pena final impuesta (años/multa) si la hay.]
- **Fundamento Legal:** [Enumera los artículos, códigos y jurisprudencias exactas citados como base de la decisión.]
- **Pruebas Decisivas:** [Enumera las pruebas cruciales que sustentaron el fallo (ej: Testimonial del ofendido, Peritaje balístico).]

Contexto (Sentencia):
{contexto}

Hechos Extraídos:
"""

PROMPT_REPORTE_PENAL_V11 = """
Con base en los siguientes HECHOS EXTRAÍDOS de una sentencia penal de segunda instancia, genera un informe legal
profesional, **detallado y analítico**, útil para un defensor que prepara o contesta una apelación.
""" + _INSTRUCCION_FORMATO + """

**Hechos Extraídos:**
{hechos}

**Formato del Informe:**
#### 1. Resumen Ejecutivo
[Resumen breve (2-3 frases): delito, acto impugnado, quién apeló y qué resolvió la Sala.]

#### 2. Datos de Identificación del Proceso
- **Tribunal:** [dato 'Tribunal']
- **Expediente:** [dato 'Identificador Judicial']
- **Etapa Procesal:** [dato 'Etapa Procesal']
- **Acto Impugnado:** [dato 'Acto Impugnado']
- **Apelante:** [dato 'Apelante']
- **Delito(s):** [dato 'Delito(s) Principal(es)']
- **Sentido del Fallo:** [dato 'Sentido del Fallo']

#### 3. Partes Involucradas
- **Acusado(s):** [dato 'Acusado(s)']
- **Víctima(s)/Ofendido(s):** [dato 'Víctima(s)/Ofendido(s)']

#### 4. Agravios y su Calificación
[Para cada agravio de 'Agravios Planteados', una viñeta con el agravio y su calificación según 'Calificación de los Agravios'.]

#### 5. Análisis de Hechos y Decisión
**Hechos Clave:** [narrativa de 'Hechos Clave']

**Decisión del Tribunal:** [dato 'Decisión Específica']

**Razonamiento del Tribunal:** [Con base en 'Fundamento Legal' y 'Pruebas Decisivas', **explica** por qué la Sala resolvió así y qué argumentos de la defensa prosperaron o no.]
"""

# v11, una sola llamada: la ficha penal completa como JSON (config.FICHA_MODO = "json"). El informe en Markdown se
# arma con código a partir del JSON, sin segunda llamada al LLM.
PROMPT_FICHA_PENAL_JSON = """
Eres un Analista Legal Penal que apoya a la DEFENSA en segunda instancia. Lee la SENTENCIA PENAL (resolución de un
recurso de apelación de una Sala Penal) y devuelve su ficha como UN objeto JSON con exactamente estas claves:

{{
  "tribunal": "Sala que resolvió",
  "fecha_resolucion": "fecha exacta de la resolución",
  "identificador": "toca / expediente / causa tal como aparece",
  "etapa_procesal": "etapa del proceso en que se dictó la resolución apelada",
  "acto_impugnado": "qué resolución se apeló (auto de vinculación a proceso, prisión preventiva, sentencia definitiva de juicio oral, sobreseimiento, otro)",
  "apelante": "quién apeló: defensa, Ministerio Público, asesor jurídico de la víctima",
  "delitos": "delito(s) imputados o acreditados, con su modalidad",
  "hechos_clave": "2 a 4 frases con la conducta y sus circunstancias de modo, tiempo y lugar",
  "agravios": [{{"agravio": "argumento del apelante, en una frase", "calificacion": "FUNDADO | INFUNDADO | INOPERANTE | NO DISPONIBLE", "razon": "por qué lo calificó así la Sala, en una frase"}}],
  "sentido_fallo": "EXACTAMENTE uno de: CONFIRMA, MODIFICA, REVOCA, REPONE, OTRO",
  "decision": "detalle del fallo (qué se confirma, modifica o revoca; pena final si la hay)",
  "fundamento_legal": ["cada artículo o jurisprudencia citada como base de la decisión, tal como aparece"],
  "pruebas_decisivas": ["cada prueba que sustentó el fallo"],
  "resumen": "2 o 3 frases: delito, acto impugnado, quién apeló y qué resolvió la Sala"
}}

Reglas:
- Usa SOLO información del texto. Si un dato no aparece o fue testado ([ELIMINADO], [No.1]...), escribe
  "[DATO NO DISPONIBLE]" (en listas, deja la lista vacía). No inventes artículos, fechas ni nombres.
- Responde ÚNICAMENTE con el JSON, en español.

Sentencia:
{contexto}
"""

# v11.2, solo para el experimento `fundamento-prompt` (no es la ficha de la app): la misma ficha con el fundamento
# legal restringido a lo que invoca la Sala para fundar SU resolución. Los artículos que la ficha precalculada cita
# fuera de ese fundamento casi siempre aparecen en la sentencia (agravios, primera instancia): no son alucinaciones
# sino artículos de otra parte del texto (conteo en la corrida). Adoptarlo exige regenerar las fichas precalculadas.
_FUNDAMENTO_ACTUAL = '  "fundamento_legal": ["cada artículo o jurisprudencia citada como base de la decisión, tal como aparece"],'
_FUNDAMENTO_SALA = ('  "fundamento_legal": ["cada artículo que la SALA invoca para fundar SU resolución (p. ej., después de '
                    '\'con fundamento en\' o \'de conformidad con\', o en los puntos resolutivos), con su ordenamiento"],')
_REGLA_FINAL = "- Responde ÚNICAMENTE con el JSON, en español."
_REGLA_FUNDAMENTO = ('- En "fundamento_legal" NO incluyas los artículos que solo menciona el apelante en sus agravios, los que '
                     'solo aplicó el juez de primera instancia ni los que la Sala cita al transcribir o resumir; si no hay, '
                     'deja la lista vacía.\n' + _REGLA_FINAL)
assert _FUNDAMENTO_ACTUAL in PROMPT_FICHA_PENAL_JSON and _REGLA_FINAL in PROMPT_FICHA_PENAL_JSON
PROMPT_FICHA_PENAL_JSON_FUNDAMENTO_SALA = PROMPT_FICHA_PENAL_JSON.replace(_FUNDAMENTO_ACTUAL, _FUNDAMENTO_SALA).replace(
    _REGLA_FINAL, _REGLA_FUNDAMENTO)

# v11.2: tercer intento de la ficha JSON. 62 de las 66 fichas inválidas de llama3.1:8b solo carecían de 'delitos'
# (o 'resumen'), aun después del reintento (fichas-diagnostico 20261116-213316_1fc6dd4). Se piden únicamente esos
# campos y se unen a la respuesta anterior; no cambia la versión de la ficha (el prompt principal es el mismo).
PROMPT_FICHA_FALTANTES = """
De la sentencia penal de apelación de abajo, responde ÚNICAMENTE con este JSON, en español:
{{
{campos}
}}

El delito siempre aparece en la sentencia (aunque los nombres de las personas estén testados): escribe el que se
imputó o se tuvo por acreditado, con su modalidad. No escribas "[DATO NO DISPONIBLE]" en estos campos.

Sentencia:
{contexto}
"""
CAMPOS_FICHA_FALTANTES = {
    "delitos": '  "delitos": "delito(s) imputados o acreditados, con su modalidad"',
    "resumen": '  "resumen": "2 o 3 frases: delito, acto impugnado, quién apeló y qué resolvió la Sala"',
}

PROMPTS_PENAL_VERSIONES = {
    "v10": (PROMPT_EXTRACCION_PENAL, PROMPT_REPORTE_PENAL),
    "v11": (PROMPT_EXTRACCION_PENAL_V11, PROMPT_REPORTE_PENAL_V11),
}


def prompts_penal(version: str | None = None) -> tuple[str, str]:
    """(extracción, informe) penales de la versión indicada (por defecto `config.PROMPT_PENAL_VERSION`)."""
    from . import config

    version = version or config.PROMPT_PENAL_VERSION
    if version not in PROMPTS_PENAL_VERSIONES:
        raise ValueError(f"Versión de prompt penal desconocida: {version}")
    return PROMPTS_PENAL_VERSIONES[version]


# --- Solo para la evaluación: borrador de consulta a partir de una sentencia (lo revisa una persona) ---
PROMPT_BORRADOR_CONSULTA = """
Eres un defensor penal en Jalisco. A partir del extracto de una sentencia de segunda instancia, escribe la consulta
que harías a un buscador de precedentes sobre TU caso, ANTES de conocer esa sentencia: en 2 a 4 frases, en primera
persona ("Mi cliente...", "Represento a..."), cuenta los hechos y qué resolución quieres apelar y por qué.
Si en la sentencia apeló el Ministerio Público o la víctima, escribe como el defensor que contesta esa apelación.

Reglas obligatorias:
- Describe un caso, no hagas preguntas generales ("¿Qué sucede cuando...?").
- NO copies frases del extracto (nunca más de 5 palabras seguidas iguales).
- NO escribas el nombre técnico del delito tal como aparece en la sentencia; describe la conducta.
- NO incluyas nombres, números de toca o carpeta, fechas, años, cantidades exactas, años o meses de pena, ni el
  resultado de la apelación.
Responde ÚNICAMENTE con el texto de la consulta.

Extracto de la sentencia:
{extracto}

Consulta:
"""

# v11.1: para regenerar los borradores que el verificador marcó (copiaban frases, nombraban el delito o traían cifras).
PROMPT_BORRADOR_CONSULTA_ESTRICTO = """
Eres un defensor penal en Jalisco. A partir del extracto de una sentencia de segunda instancia, escribe la consulta
que harías a un buscador de precedentes sobre TU caso, ANTES de conocer esa sentencia: en 2 a 4 frases, en primera
persona ("Mi cliente...", "Represento a..."), cuenta qué hizo o qué se le atribuye a tu cliente y qué resolución
quieres apelar y por qué. Si apeló el Ministerio Público o la víctima, escribe como el defensor que contesta.

Reglas obligatorias (si rompes una, la consulta se descarta):
- PALABRAS PROHIBIDAS: {prohibidas}. Describe la conducta con otras palabras.
- NINGÚN número, ni con cifras ni con letra: sin fechas, años, cantidades, montos, números de toca ni duraciones
  de pena.
- Escribe con tus palabras: ni una frase del extracto, ni siquiera 5 palabras seguidas iguales.
- Sin nombres de personas, lugares, juzgados ni el resultado de la apelación.
Responde ÚNICAMENTE con el texto de la consulta.

Extracto de la sentencia:
{extracto}

Consulta:
"""

PROMPT_BORRADOR_SIN_RESPUESTA = """
Eres un defensor penal en Jalisco. Escribe en 2 a 4 frases la consulta que harías a un buscador de precedentes sobre
este tema: "{tema}". En primera persona ("Mi cliente..."), describe un caso concreto y verosímil (hechos y qué
resolución quieres apelar), sin nombres, fechas, años ni cantidades exactas.
Responde ÚNICAMENTE con el texto de la consulta.

Consulta:
"""


# --------------------------------------------------------------------------------------------------------------
# Jueces LLM de la evaluación automática (v11.2). Los usa un modelo DISTINTO del que genera las fichas; los criterios
# son los de la guía de anotación (evaluacion/README.md). Responden solo JSON con valores cerrados.
# --------------------------------------------------------------------------------------------------------------

PROMPT_JUEZ_SENTENCIA = """
Lees una sentencia de apelación penal del Supremo Tribunal de Justicia de Jalisco. El texto viene de un PDF: está en
minúsculas, a veces sin acentos o con palabras partidas. Responde SOLO con lo que diga el texto; si no lo dice,
escribe "no determinado".

1. familia: el delito principal; elige una de estas familias: {familias}. Los delitos cometidos por culpa o
   imprudencia (homicidio culposo, lesiones o daños a título de culpa, p. ej. en un accidente de tránsito) son
   "delitos culposos". Si la sentencia trata varios delitos, elige el que se menciona primero.
2. acto_impugnado: la resolución que se apeló: "vinculación" (auto de vinculación a proceso), "no vinculación" (auto
   que niega la vinculación), "medida cautelar" (prisión preventiva u otra medida cautelar), "sobreseimiento",
   "sentencia definitiva" (condena o absolución, también en procedimiento abreviado) u "otro" para cualquier otra
   resolución (p. ej. orden de aprehensión o su negativa, denegada apelación, reparación del daño, auto de formal
   prisión del sistema tradicional, beneficios o sustitutivos de la pena).
3. apelante: quién apeló: "defensa" (el imputado, el sentenciado o su defensor), "ministerio público", "asesor de la
   víctima" (la víctima, el ofendido o su asesor) o "varios" (más de una parte).
4. sentido: lo que resolvió la Sala en los puntos resolutivos: "CONFIRMA", "MODIFICA", "REVOCA" o "REPONE" (ordena
   reponer el procedimiento).

Responde únicamente con JSON:
{{"familia": "...", "acto_impugnado": "...", "apelante": "...", "sentido": "..."}}

Inicio de la sentencia:
{inicio}

Puntos resolutivos (parte final de la sentencia):
{final}
"""

PROMPT_JUEZ_RELEVANCIA = """
Consulta de un defensor penal sobre su caso:
{consulta}

Sentencia de una Sala Penal, categoría "{categoria}" en el portal del tribunal (inicio y puntos resolutivos):
{sentencia}

Eres un abogado penalista y evalúas si esa sentencia le sirve al defensor como precedente. Contesta en este orden:
1. delito_consulta: el delito que se atribuye en la consulta, en pocas palabras.
2. delito_sentencia: el delito de la sentencia, en pocas palabras.
3. mismo_delito: "si" si son el mismo delito o de la misma familia (p. ej. robo calificado y robo equiparado; abuso
   sexual y violación), "no" si son delitos distintos.
4. acto_consulta y acto_sentencia: qué resolución se apela en cada una (vinculación a proceso, no vinculación,
   prisión preventiva u otra medida cautelar, sentencia definitiva, sobreseimiento u otra).
5. mismo_problema_procesal: "si" si se apela el mismo tipo de resolución o se discute el mismo argumento; "no" si no.
6. relevancia: 2 = mismo delito y contexto compatible, útil como precedente directo; 1 = misma familia de delito, o
   mismo problema procesal con hechos distintos; 0 = otro delito, o solo coinciden palabras.

Responde únicamente con JSON con esas claves:
{{"delito_consulta": "", "delito_sentencia": "", "mismo_delito": "", "acto_consulta": "", "acto_sentencia": "",
"mismo_problema_procesal": "", "relevancia": 0}}
"""

PROMPT_JUEZ_CONSULTA = """
Revisas consultas de prueba para un buscador de precedentes penales. Cada una debe parecer escrita por un defensor
sobre un caso concreto. Contesta:

- es_consulta: "si" si es una consulta coherente en español que describe lo que se atribuye a una persona y qué
  resolución se quiere apelar o qué se busca; "no" si es otra cosa (una negativa, una respuesta, una instrucción,
  texto sin sentido o sin un caso concreto).
- delito_identificable: "si" si un abogado penalista sabría de qué delito se trata al leerla; "no" si no.

Responde únicamente con JSON:
{{"es_consulta": "si|no", "delito_identificable": "si|no"}}

Consulta:
{consulta}
"""

# Tercer intento del juez de relevancia: mistral:7b devolvía el JSON completo menos la clave "relevancia"
# (1,293 de 3,381 pares). Se le pide solo esa clave, con su propia respuesta anterior a la vista.
PROMPT_JUEZ_RELEVANCIA_FALTANTE = """{prompt}
Tu respuesta anterior no traía la calificación final:
{previa}

Responde únicamente con JSON con la clave que falta (2, 1 o 0, según la escala de arriba):
{{"relevancia": 0}}
"""
