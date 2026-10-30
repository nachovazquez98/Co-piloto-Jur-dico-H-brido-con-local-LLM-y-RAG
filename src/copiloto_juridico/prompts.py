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

PROMPT_EXTRACCION_CIVIL = """
Eres un Analista Legal Civil. Tu misión es extraer la información más exhaustiva y crítica de la SENTENCIA CIVIL.

""" + _NOTA_ANONIMIZACION + """

**Formato de Extracción (Detallado):**
- **Tribunal:** [Nombre completo del Juzgado/Sala que resolvió, incluyendo la fecha exacta de la resolución]
- **Identificador Judicial:** [Número de Expediente o Toca, tal como aparece]
- **Etapa Procesal:** [Ej: Sentencia Definitiva de Segunda Instancia, Interlocutoria]
- **Acción Civil Principal:** [Acción ejercida de forma completa (ej. Acción Reivindicatoria, Rescisión de Contrato de Arrendamiento).]
- **Parte Actora:** [Nombre completo o identificador de quien demandó]
- **Parte Demandada:** [Nombre completo o identificador de quien fue demandado]
- **Objeto del Litigio:** [Describe el bien o derecho específico en disputa (ej. Inmueble con clave catastral X).]
- **Hechos Clave:** [Narrativa breve y fáctica de la disputa que llevó a la demanda.]
- **Decisión Específica:** [Fallo exacto: Procede/No procede la acción, Condena/Absuelve a la parte demandada, cuantía económica principal.]
- **Fundamento Legal:** [Enumera los artículos, códigos y jurisprudencias exactas citados como base de la decisión.]
- **Pruebas Decisivas:** [Enumera las pruebas documentales, periciales o testimoniales que fueron clave para la decisión.]

Contexto (Sentencia):
{contexto}

Hechos Extraídos:
"""

PROMPT_EXTRACCION_FAMILIAR = """
Eres un Analista Legal Familiar. Tu misión es extraer la información más exhaustiva y crítica de la SENTENCIA FAMILIAR.

""" + _NOTA_ANONIMIZACION + """

**Formato de Extracción (Detallado):**
- **Tribunal:** [Nombre completo del Juzgado/Sala Familiar que resolvió, incluyendo la fecha exacta de la resolución]
- **Identificador Judicial:** [Número de Expediente]
- **Etapa Procesal:** [Ej: Sentencia Definitiva de Primera Instancia, Resolución de Incidente de Modificación]
- **Materia Principal:** [Tema Familiar específico (ej. Divorcio Necesario, Modificación de Guarda y Custodia).]
- **Progenitor Actor:** [Nombre de quien demanda]
- **Progenitor Demandado:** [Nombre del demandado]
- **Menores Involucrados:** [Lista de nombres o identificadores de los hijos menores de edad.]
- **Hechos Clave:** [Narrativa breve de la situación familiar que generó el conflicto.]
- **Decisión Específica:** [Fallo exacto sobre Custodia, Pensión (monto o porcentaje) y Régimen de Convivencias.]
- **Fundamento Clave:** [Describe el razonamiento clave del juez para proteger al menor, citando artículos de leyes familiares.]
- **Pruebas Decisivas:** [Enumera las pruebas cruciales: Estudios Socioeconómicos, Dictámenes Psicológicos, Opinión del Menor, etc.]

Contexto (Sentencia):
{contexto}

Hechos Extraídos:
"""

PROMPT_EXTRACCION_MERCANTIL = """
Eres un Analista Legal Mercantil. Tu misión es extraer la información más exhaustiva y crítica de la RESOLUCIÓN MERCANTIL.

""" + _NOTA_ANONIMIZACION + """

**Formato de Extracción (Detallado):**
- **Tribunal:** [Nombre completo del Juzgado/Sala que resolvió, incluyendo la fecha exacta de la resolución]
- **Identificador Judicial:** [Número de Expediente/Toca]
- **Etapa Procesal:** [Ej: Sentencia Interlocutoria, Juicio Ejecutivo Mercantil]
- **Acción Principal:** [Materia Mercantil específica (ej. "Acción Cambiaria Directa", "Cumplimiento de Contrato").]
- **Parte Actora (Acreedora):** [Nombre de quien demanda el pago]
- **Parte Demandada (Deudora):** [Nombre de quien debe]
- **Documento Base:** [Tipo de documento que originó la deuda (ej. Pagaré, Cheque Devuelto, Contrato de Suministro).]
- **Hechos Clave:** [Narrativa breve de la disputa, el monto de la deuda original y la razón del incumplimiento.]
- **Decisión Específica:** [Fallo exacto: Condena al pago, Absolución, Se declara la disolución de la sociedad.]
- **Consecuencias Financieras:** [Monto exacto a pagar (suerte principal), intereses (ordinarios/moratorios) y condena en costas.]
- **Fundamento Legal:** [Enumera los artículos, códigos (LGTOC, Código de Comercio) y jurisprudencias citados como base del fallo.]
- **Pruebas Decisivas:** [Enumera las pruebas cruciales (ej. El pagaré endosado, Certificación de Estado de Cuenta, Peritaje Contable).]

Contexto (Sentencia):
{contexto}

Hechos Extraídos:
"""

PROMPT_EXTRACCION_DEFAULT = """
Tu tarea es analizar el siguiente texto de una sentencia judicial y extraer los hechos clave de forma estructurada.

""" + _NOTA_ANONIMIZACION + """

**Formato de Extracción:**
- **Tribunal:** [Fecha y nombre completo del Tribunal/Juzgado]
- **Expediente:** [Número de Expediente/Toca]
- **Materia:** [Tema principal del litigio]
- **Partes:** [Identificación de las partes si se mencionan]
- **Hechos Clave:** [Narrativa breve de lo que ocurrió]
- **Decisión Específica:** [Fallo principal (ej. "Se condena", "Se absuelve", "Se modifica")]
- **Fundamentos/Pruebas:** [Razón principal de la decisión o pruebas clave, si se mencionan]
- **Leyes Citadas:** [Artículos de ley mencionados, si los hay]

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

PROMPT_REPORTE_CIVIL = """
Con base en los siguientes HECHOS EXTRAÍDOS de una sentencia civil, genera un informe legal profesional, **detallado y analítico**.
Utiliza los 'Hechos Extraídos' para construir una narrativa coherente y explicar las razones del tribunal.
""" + _INSTRUCCION_FORMATO + """

**Hechos Extraídos:**
{hechos}

**Formato del Informe:**
#### 1. Resumen Ejecutivo
[Resumen breve (2-3 frases) de la acción principal, el objeto del litigio y quién ganó el juicio.]

#### 2. Datos de Identificación del Proceso
- **Tribunal:** [dato 'Tribunal']
- **Expediente:** [dato 'Identificador Judicial']
- **Etapa Procesal:** [dato 'Etapa Procesal']
- **Acción Principal:** [dato 'Acción Civil Principal']

#### 3. Partes y Objeto del Litigio
- **Parte Actora:** [dato 'Parte Actora']
- **Parte Demandada:** [dato 'Parte Demandada']
- **Objeto del Litigio:** [dato 'Objeto del Litigio']

#### 4. Análisis de Hechos y Decisión
**Hechos Clave:** [narrativa de 'Hechos Clave']

**Decisión del Tribunal:** [dato 'Decisión Específica']

**Razonamiento del Tribunal:** [Con base en 'Fundamento Legal' y 'Pruebas Decisivas', **explica** por qué la acción procedió o no. ¿Qué parte probó sus afirmaciones y cómo?]
"""

PROMPT_REPORTE_FAMILIAR = """
Con base en los siguientes HECHOS EXTRAÍDOS de una sentencia familiar, genera un informe legal profesional, **detallado y analítico**.
Utiliza los 'Hechos Extraídos' para construir una narrativa coherente, **enfocándote en el Interés Superior del Menor** cuando aplique.
""" + _INSTRUCCION_FORMATO + """

**Hechos Extraídos:**
{hechos}

**Formato del Informe:**
#### 1. Resumen Ejecutivo
[Resumen breve (2-3 frases) del conflicto familiar y la decisión final (custodia, alimentos, divorcio, etc.).]

#### 2. Datos de Identificación del Proceso
- **Tribunal:** [dato 'Tribunal']
- **Expediente:** [dato 'Identificador Judicial']
- **Etapa Procesal:** [dato 'Etapa Procesal']
- **Materia Principal:** [dato 'Materia Principal']

#### 3. Partes y Menores Involucrados
- **Parte Actora:** [dato 'Progenitor Actor']
- **Parte Demandada:** [dato 'Progenitor Demandado']
- **Menores:** [dato 'Menores Involucrados']

#### 4. Decisión y Fundamentos
**Decisión Específica:** [dato 'Decisión Específica']

**Razonamiento del Tribunal:** [Con base en 'Fundamento Clave' y 'Pruebas Decisivas', **explica** por qué el juez tomó esa decisión y cómo se protegió el Interés Superior del Menor, si aplica.]
"""

PROMPT_REPORTE_MERCANTIL = """
Con base en los siguientes HECHOS EXTRAÍDOS de una resolución mercantil, genera un informe legal profesional, **detallado y analítico**.
Utiliza los 'Hechos Extraídos' para construir una narrativa coherente y explicar las consecuencias económicas.
""" + _INSTRUCCION_FORMATO + """

**Hechos Extraídos:**
{hechos}

**Formato del Informe:**
#### 1. Resumen Ejecutivo
[Resumen breve (2-3 frases) de la acción mercantil, el documento base y el resultado (si se ordenó el pago o no).]

#### 2. Datos del Proceso
- **Tribunal:** [dato 'Tribunal']
- **Expediente:** [dato 'Identificador Judicial']
- **Etapa Procesal:** [dato 'Etapa Procesal']
- **Acción Principal:** [dato 'Acción Principal']

#### 3. Partes y Documento Base
- **Parte Actora (Acreedora):** [dato 'Parte Actora (Acreedora)']
- **Parte Demandada (Deudora):** [dato 'Parte Demandada (Deudora)']
- **Documento Base:** [dato 'Documento Base']

#### 4. Hechos y Decisión
**Hechos Clave:** [narrativa de 'Hechos Clave']

**Decisión Final:** [dato 'Decisión Específica']

#### 5. Análisis y Consecuencias
**Razonamiento del Tribunal:** [Con base en 'Fundamento Legal' y 'Pruebas Decisivas', **explica** por qué se condenó o absolvió al deudor.]

**Consecuencias Financieras:** [dato 'Consecuencias Financieras']
"""

PROMPT_REPORTE_DEFAULT = """
Con base en los siguientes HECHOS EXTRAÍDOS, genera un informe legal estructurado y **analítico**.
Utiliza los 'Hechos Extraídos' para construir una narrativa coherente.
""" + _INSTRUCCION_FORMATO + """

**Hechos Extraídos:**
{hechos}

**Formato del Informe:**
#### 1. Resumen de la Decisión
[Resumen breve del tema principal y la decisión final.]

#### 2. Detalles del Caso
- **Materia:** [dato 'Materia']
- **Tribunal:** [dato 'Tribunal']
- **Expediente:** [dato 'Expediente']
- **Partes:** [dato 'Partes']

#### 3. Hechos y Decisión
**Hechos Clave:** [narrativa de 'Hechos Clave']

**Decisión Final:** [dato 'Decisión Específica']

#### 4. Fundamentos del Tribunal
[**Explica** brevemente por qué el tribunal tomó esa decisión, con base en 'Fundamentos/Pruebas' y 'Leyes Citadas'.]
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

PROMPTS_POR_MATERIA = {
    "penal": (PROMPT_EXTRACCION_PENAL, PROMPT_REPORTE_PENAL),
    "civil": (PROMPT_EXTRACCION_CIVIL, PROMPT_REPORTE_CIVIL),
    "familiar": (PROMPT_EXTRACCION_FAMILIAR, PROMPT_REPORTE_FAMILIAR),
    "mercantil": (PROMPT_EXTRACCION_MERCANTIL, PROMPT_REPORTE_MERCANTIL),
}
PROMPTS_DEFAULT = (PROMPT_EXTRACCION_DEFAULT, PROMPT_REPORTE_DEFAULT)
