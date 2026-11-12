"""Textos de ayuda de la interfaz: qué significa cada número y cada término (sección 7 del informe).

Se centralizan aquí para que la app, la página "Cómo funciona" y la página "Evaluación" usen exactamente las
mismas definiciones (una sola versión de cada concepto).
"""

from . import config

# --- Números que muestra la app (qué significan y qué NO significan) ---
AYUDA = {
    "similitud": ("Coseno entre la consulta y el fragmento más parecido de la sentencia. Con este modelo de "
                  "embeddings casi todo cae entre 0.75 y 0.90: sirve para ordenar, no es un porcentaje de parecido "
                  "ni una probabilidad de relevancia."),
    "rrf": (f"Posición combinada en la búsqueda semántica (embeddings) y léxica (BM25): Σ 1/(k + rango), con "
            f"k = {config.RRF_K}. Máximo {2 / config.RRF_K:.3f} (primer lugar en ambas); un documento que solo aparece "
            "en una lista vale como máximo la mitad. No es una probabilidad ni se compara entre consultas."),
    "score_hibrido": ("Puntaje para ordenar las categorías dentro de esta consulta: suma ponderada de coseno, "
                      "Jaccard, KEM, concisión y voto, multiplicada por las penalizaciones. No es una probabilidad: "
                      "0.62 no significa 62 % de confianza."),
    "coseno": ("Coseno entre la consulta y el NOMBRE de la categoría (no las sentencias). Típicamente 0.75–0.90. "
               "En el score se reescala al rango de los candidatos de esta consulta."),
    "jaccard": ("Palabras (lemas) en común entre los conceptos clave y el nombre de la categoría, divididas entre "
                "todas las palabras de ambos: |A∩B| / |A∪B|. Castiga nombres largos."),
    "kem": ("Key Entity Match: fracción de los términos importantes de la consulta que aparecen en el nombre de la "
            "categoría: |A∩B| / |A|. No castiga que la categoría tenga palabras extra."),
    "concision": "1/√(número de lemas del nombre de la categoría): favorece nombres generales sobre muy específicos.",
    "voto": ("Cuánto aparecen sentencias de esta categoría en la búsqueda amplia (suma de su RRF), dividido entre la "
             "categoría más votada: 1 = la más representada entre las sentencias recuperadas."),
    "tokens": "Número de lemas del nombre de la categoría (entra en la concisión y en la penalización por longitud).",
    "penalidad": ("Multiplicador del score: ×0.90 si KEM < 0.6 y el nombre tiene más de 5 lemas; ×0.70 si el nombre "
                  "contiene una palabra de una lista fija. 1 = sin penalización."),
    "descartados": ("Candidatos eliminados por los controles de calidad (QC_1, extracción, QC_2). Muchos descartes "
                    "indican una consulta ambigua o un tema poco cubierto; no son, por sí mismos, un error."),
    "precedentes_validos": "Fichas que superaron todos los controles y entran en la síntesis.",
    "ambiguos": ("Veces que QC_1 o QC_2 respondieron algo que no se pudo leer como SI/NO. En ese caso el sistema "
                 "continúa (falla abierta), así que esa ficha tuvo un control menos."),
    "tiempo": "Tiempo total de la consulta; la mayor parte son llamadas al LLM (≈ 4 por candidato aceptado).",
    "modelo": "LLM que redactó las fichas y la síntesis.",
    "fuente": ("'Filtro estricto' = la sentencia es de una de las 3 categorías detectadas; 'Búsqueda amplia' = se "
               "añadió sin filtro porque el filtro dio pocos candidatos."),
    "sentido_fallo": "Resultado de la apelación: CONFIRMA, MODIFICA o REVOCA la resolución apelada; REPONE el procedimiento; u OTRO.",
    "acto_impugnado": "La resolución que se apeló (p. ej. auto de vinculación a proceso, prisión preventiva, sentencia de juicio oral).",
    "agravios": "Argumentos del apelante contra la resolución y cómo los calificó la Sala (fundados, infundados, inoperantes).",
}

# --- Métricas de evaluación (sección 5) ---
METRICAS = {
    "Hit@3 tipo": ("Fracción de consultas cuyo tipo correcto queda dentro del filtro de 3 categorías. Si queda fuera, "
                   "ninguna etapa posterior puede recuperarlo."),
    "Hit@3 familia": "Igual, pero basta con acertar la familia de delito (p. ej. 'robo' en vez de 'robo calificado').",
    "Top-1 tipo": "Exactitud: la primera categoría es el tipo correcto.",
    "Top-1 familia": "La primera categoría es de la familia correcta.",
    "F1 macro": ("Promedio del F1 de cada clase (todas pesan igual), con F1 = 2·P·R/(P+R). Evita que los delitos "
                 "frecuentes dominen el resultado."),
    "Hit@k": "Fracción de consultas en las que la sentencia buscada aparece entre los k primeros resultados.",
    "MRR": ("Promedio de 1/rango del primer acierto (0 si no aparece). 0.5 ≈ en promedio el acierto está en 2.º lugar."),
    "nDCG@5": ("DCG@5 / DCG ideal@5 con DCG = Σ rel_i / log2(i+1) y relevancia 0/1/2. 1 = orden perfecto; mide si "
               "lo más relevante aparece primero."),
    "P@5": "De los 5 primeros resultados, fracción relevante.",
    "precisión": "De lo que el control deja pasar, cuánto es realmente relevante: VP / (VP + FP).",
    "recall": ("De lo realmente relevante, cuánto conserva el control: VP / (VP + FN). Para la defensa es el criterio "
               "principal: un falso negativo es un precedente útil perdido."),
    "especificidad": "De lo irrelevante, cuánto elimina el control: VN / (VN + FP).",
    "kappa": "Acuerdo con el humano descontando el acuerdo por azar: (p_o − p_e)/(1 − p_e). 0.61–0.80 = sustancial.",
    "IC95": ("Intervalo de confianza al 95 %: rango plausible del valor si se hubieran escrito otras consultas "
             "parecidas. Wilson para proporciones; bootstrap con 1,000 remuestreos para promedios."),
    "Δ": ("Diferencia pareada entre dos variantes sobre las mismas consultas, con IC bootstrap. Una mejora se acepta "
          "solo si su intervalo no incluye 0."),
    "p50 / p95": "Mediana (experiencia típica) y percentil 95 (la peor razonable) de la latencia.",
}

# --- Glosario jurídico (sección 2.3; definiciones operativas a validar con el asesor jurídico) ---
GLOSARIO = {
    "Toca": "Expediente que abre el tribunal de segunda instancia para resolver un recurso. Cada documento del corpus "
            "es la resolución de un toca.",
    "Recurso de apelación": "Medio por el que una parte (defensa, Ministerio Público, asesor de la víctima) pide a la "
                            "Sala revisar una decisión del juez de primera instancia.",
    "Agravios": "Argumentos con los que el apelante explica por qué la resolución le perjudica. La Sala los califica "
                "como fundados, infundados o inoperantes. Es el dato más útil para el defensor.",
    "Acto impugnado": "Lo que se apela: auto de vinculación o no vinculación a proceso, medida cautelar (p. ej. "
                      "prisión preventiva), sentencia de juicio oral, sobreseimiento.",
    "Sentido del fallo": "Resultado de la apelación: confirma, modifica o revoca la resolución apelada; puede ordenar "
                         "reponer el procedimiento.",
    "Fundamento legal": "Artículos citados: principalmente Código Penal del Estado de Jalisco (tipo penal) y Código "
                        "Nacional de Procedimientos Penales (proceso).",
    "Precedente vs. jurisprudencia": "Las resoluciones del STJ no son jurisprudencia obligatoria; muestran el criterio "
                                     "que la Sala ha aplicado y sirven como argumento orientador.",
    "Datos testados": "El STJ elimina datos personales ([ELIMINADO], [No.1]…). El sistema los reporta como 'dato no "
                      "disponible', nunca los infiere.",
}

USO_RESPONSABLE = (
    "Herramienta de apoyo para encontrar y leer precedentes públicos del STJ Jalisco. No predice el resultado de un "
    "juicio, no redacta agravios y no sustituye la lectura de la sentencia original ni el criterio del abogado. "
    "Verifica siempre la ficha contra la sentencia (identificador del documento) antes de citarla: el LLM puede "
    "omitir datos o equivocarse, y los controles de calidad reducen ese riesgo pero no lo eliminan.")
