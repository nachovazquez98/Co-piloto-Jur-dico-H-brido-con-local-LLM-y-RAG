"""Textos de ayuda de la interfaz: una frase por número que ve el usuario, y el glosario jurídico breve.

Se centralizan aquí para que la pantalla principal y "Cómo funciona" usen exactamente las mismas definiciones.
"""

from . import config

AYUDA = {
    "similitud": ("Coseno entre la consulta y el fragmento más parecido de la sentencia; con este modelo casi todo "
                  "cae entre 0.75 y 0.90: sirve para ordenar, no es un porcentaje."),
    "rrf": (f"Posición combinada en la búsqueda semántica y la léxica (BM25); máximo {2 / config.RRF_K:.3f} = primero "
            "en ambas. No es una probabilidad."),
    "score_hibrido": "Puntaje para ordenar las categorías dentro de esta consulta; no es una probabilidad.",
    "coseno": "Parecido entre la consulta y el nombre de la categoría (embeddings).",
    "jaccard": "Palabras en común entre los conceptos clave y el nombre de la categoría, sobre el total de palabras.",
    "voto": "Cuánto aparecen sentencias de esa categoría entre las recuperadas (1 = la más representada).",
    "fuente": ("Filtro estricto: sentencia de los delitos seleccionados o, sin selección, de una de las 3 familias "
               "detectadas. Búsqueda amplia: añadida sin filtro porque el filtro arrojó pocos candidatos."),
    "delito": ("Opcional. Sin selección, el sistema deduce el delito de la descripción y busca en las 3 familias más "
               "probables. Con uno o varios delitos seleccionados, solo se buscan sentencias de esos delitos (si son "
               "pocas, se completa con las más parecidas de otros delitos, identificadas como búsqueda amplia). "
               "Entre paréntesis, el número de sentencias disponibles de cada uno."),
    "descartados": "Candidatos eliminados por los controles; muchos descartes indican una consulta ambigua o un tema poco cubierto.",
    "ambiguos": "El control respondió algo que no se pudo leer como SI/NO y el sistema continuó.",
    "sentido_fallo": ("Leído de los puntos resolutivos de la sentencia cuando es posible; 'verificar' = la ficha del "
                      "modelo dice otra cosa."),
    "parte_favorecida": ("Parte que obtuvo la resolución de la Sala, deducida del apelante y del sentido del fallo: "
                         "si apeló la defensa y la Sala confirmó, la acusación; si revocó, el acusado."),
}

GLOSARIO = {
    "Recurso de apelación": ("Medio de impugnación por el que la parte inconforme solicita a la Sala Penal que "
                             "confirme, modifique o revoque una resolución del juez de primera instancia."),
    "Toca": "Expediente que abre la Sala para resolver un recurso. Cada documento del corpus es la resolución de un toca.",
    "Acto impugnado": ("Resolución que se apela: auto de vinculación o de no vinculación a proceso, medida cautelar "
                       "(p. ej. prisión preventiva), sobreseimiento o sentencia definitiva."),
    "Apelante": "Parte que interpone el recurso: la defensa, el Ministerio Público o la víctima u ofendido (o su asesor jurídico).",
    "Agravios": ("Argumentos del apelante contra la resolución impugnada. La Sala los califica como fundados (le asiste "
                 "la razón), infundados (carecen de razón) o inoperantes (no pueden analizarse o no combaten las "
                 "consideraciones de la resolución)."),
    "Puntos resolutivos": "Parte final de la sentencia que contiene la decisión de la Sala; de ahí se lee el sentido del fallo.",
    "Sentido del fallo": ("Resultado del recurso: la Sala confirma, modifica o revoca la resolución apelada, u ordena "
                          "reponer el procedimiento."),
    "Familia de delito": ("Agrupación de los tipos penales que registra el portal del STJ (p. ej. robo calificado y robo "
                          "equiparado pertenecen a la familia robo). La familia «otros» reúne los tipos que no "
                          "encuadran en ninguna de las demás."),
    "Precedente y jurisprudencia": ("Las resoluciones del STJ no constituyen jurisprudencia obligatoria: muestran el "
                                    "criterio aplicado por la Sala y sirven como argumento orientador."),
    "Datos testados": "Datos personales suprimidos por el STJ en la versión pública ([eliminado]); el sistema nunca los infiere.",
}

# Fundamentos técnicos que explica la página "Cómo funciona" (lenguaje para el usuario, sin fórmulas).
FUNDAMENTOS = {
    "Fragmentos y embeddings": (
        "Cada sentencia se divide en fragmentos de 1,500 caracteres (alrededor de 400 palabras-unidad o «tokens»), "
        "con un traslape de 200 caracteres para no partir una idea en la frontera. El modelo que representa el "
        "significado solo admite 512 tokens por texto; lo que excede ese límite se perdería. Cada fragmento se "
        "convierte en un *embedding*: una lista de 768 números que resume su significado, de modo que dos textos "
        "que dicen lo mismo con palabras distintas quedan cercanos. El modelo es `multilingual-e5-base`, entrenado "
        "para español y otros idiomas, y lo bastante ligero para ejecutarse en el mismo equipo que el modelo de "
        "lenguaje. Las sentencias penales se indexan completas."),
    "Base vectorial (ChromaDB)": (
        "Los embeddings se guardan en una base vectorial local (ChromaDB) con un índice HNSW, que permite encontrar "
        "en milisegundos los fragmentos más cercanos a la consulta entre más de 200 mil, y filtrar por delito. Se "
        "construye en el primer arranque a partir del índice incluido en el repositorio y no requiere servidor."),
    "Búsqueda literal (BM25)": (
        "En paralelo se ejecuta una búsqueda por palabras (BM25): otorga más peso a los términos que aparecen en la "
        "sentencia y que son poco frecuentes en el resto del corpus. Complementa a la búsqueda semántica porque "
        "respeta coincidencias exactas, como el nombre del delito («robo calificado» frente a «robo equiparado») o "
        "una figura procesal."),
    "Fusión de resultados (RRF)": (
        "Las dos listas (semántica y literal) se combinan con *Reciprocal Rank Fusion*: cada sentencia suma "
        "1 / (60 + su posición) en cada lista. Se usa la posición y no el puntaje porque ambas búsquedas miden en "
        "escalas distintas; una sentencia que aparece alta en las dos es la más confiable."),
    "Clasificación por familia de delito": (
        "Antes de buscar, el sistema estima a qué familia de delito corresponde la consulta combinando la similitud "
        "semántica con el nombre de cada familia, las palabras en común y el delito de las sentencias más parecidas. "
        "La búsqueda se concentra en las 3 familias más probables, o en los delitos que seleccione el usuario."),
    "Controles de calidad": (
        "QC_1: el modelo de lenguaje verifica que cada candidata trate el mismo delito o el mismo problema procesal. "
        "Ficha: el modelo extrae, solo del texto de la resolución, los datos del recurso en un formato fijo. "
        "Verificaciones automáticas: el sentido del fallo se contrasta con los puntos resolutivos, cada artículo "
        "citado se busca en la sentencia y QC_2 comprueba que el delito de la ficha corresponda a las familias "
        "buscadas. Las discrepancias se señalan con **verificar**."),
    "Modelos y confidencialidad": (
        "El modelo de lenguaje (`llama3.1:8b`) y el de embeddings se ejecutan en el propio equipo mediante Ollama: "
        "la descripción del caso no se envía a ningún servicio externo, lo que resguarda el secreto profesional. "
        "Las respuestas son reproducibles (temperatura 0 y semilla fija). Las fichas se calculan de antemano; las "
        "pocas que el modelo principal no completó las elabora un modelo de respaldo (`qwen2.5:7b`), y el registro "
        "del análisis indica con qué modelo se hizo cada una."),
}

AVISO_LEGAL = ("Herramienta académica de apoyo basada en sentencias públicas del Supremo Tribunal de Justicia del "
               "Estado de Jalisco. No constituye asesoría jurídica ni sustituye la lectura de la resolución original "
               "ni el criterio profesional del abogado.")
