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
    "fuente": ("Filtro estricto = de los delitos que elegiste o, si no elegiste, de una de las 3 categorías detectadas; "
               "búsqueda amplia = añadida sin filtro porque el filtro dio pocos candidatos."),
    "delito": ("Opcional. Vacío: la app deduce el delito de la descripción y busca en las 3 categorías más probables. "
               "Si eliges uno o varios, solo se buscan sentencias de esos delitos (si hay pocas, se completa con las "
               "más parecidas de otros delitos y se marcan). Entre paréntesis, cuántas sentencias hay de cada uno."),
    "descartados": "Candidatos eliminados por los controles; muchos descartes indican una consulta ambigua o un tema poco cubierto.",
    "ambiguos": "El control respondió algo que no se pudo leer como SI/NO y el sistema continuó.",
    "sentido_fallo": ("Leído de los puntos resolutivos de la sentencia cuando es posible; 'verificar' = la ficha del "
                      "modelo dice otra cosa."),
    "para_defensa": ("Se deduce de quién apeló y del sentido del fallo: si apeló la defensa y la Sala confirmó, es "
                     "desfavorable; si apeló el Ministerio Público y la Sala confirmó, es favorable."),
}

GLOSARIO = {
    "Toca": "Expediente que abre la Sala para resolver un recurso. Cada documento del corpus es la resolución de un toca.",
    "Agravios": "Argumentos del apelante contra la resolución. La Sala los califica como fundados, infundados o inoperantes.",
    "Acto impugnado": "Lo que se apela: auto de vinculación a proceso, medida cautelar, sentencia de juicio oral, sobreseimiento.",
    "Sentido del fallo": "Resultado: la Sala confirma, modifica o revoca la resolución apelada, o repone el procedimiento.",
    "Precedente vs. jurisprudencia": ("Las resoluciones del STJ no son jurisprudencia obligatoria: muestran el criterio "
                                      "que ha aplicado la Sala y sirven como argumento orientador."),
    "Datos testados": "Datos personales eliminados por el STJ ([eliminado]); el sistema nunca los infiere.",
}

AVISO_LEGAL = ("Herramienta de apoyo basada en sentencias públicas del STJ Jalisco: no sustituye la lectura de la "
               "sentencia original ni el criterio del abogado.")
