"""Consultas de ejemplo para la interfaz.

- `EJEMPLOS_PENALES` (v11): casos redactados como los plantearía un defensor que prepara o contesta una
  apelación ante las Salas Penales del STJ, con distintos delitos y actos impugnados. `tipo_corpus` es el tipo
  del corpus que debería encontrarse; si es None, el tema no tiene tipo en el corpus y la app debería avisarlo.
- Las 18 consultas multimateria de v10 ya no se muestran en la app; están en `evaluacion/regresion.py`.

Estos ejemplos son para demostrar la app: **no** forman parte del conjunto de evaluación (que debe redactarse
de forma independiente, ver `evaluacion/README.md`).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ejemplo:
    materia: str
    tipo: str
    texto: str
    etapa: str | None = None          # acto impugnado o etapa del caso
    tipo_corpus: str | None = ""      # "" = no verificado (v10); None = sin tipo en el corpus

    @property
    def sin_tipo_en_corpus(self) -> bool:
        return self.tipo_corpus is None

    @property
    def etiqueta(self) -> str:
        partes = [self.materia, self.tipo] + ([self.etapa] if self.etapa else [])
        return " · ".join(partes) + (" (sin tipo en el corpus)" if self.sin_tipo_en_corpus else "")


EJEMPLOS_PENALES: list[Ejemplo] = [
    Ejemplo("Penal", "Robo calificado",
            "Mi defendido fue vinculado a proceso porque supuestamente le arrebató el celular a una mujer en una parada "
            "de camión, amagándola con un cuchillo. La única prueba es el señalamiento de la víctima en un "
            "reconocimiento por fotografía hecho sin presencia de defensor. Quiero apelar el auto de vinculación.",
            "Vinculación a proceso", "penal/robo-calificado"),
    Ejemplo("Penal", "Robo equiparado",
            "Condenaron a mi cliente en juicio oral por tener en su taller autopartes de un vehículo con reporte de robo. "
            "Él sostiene que las compró sin saber su origen y presentó notas de compra. Busco precedentes sobre cómo "
            "valora la Sala el conocimiento del origen ilícito para apelar la sentencia.",
            "Sentencia de juicio oral", "penal/robo-equiparado"),
    Ejemplo("Penal", "Homicidio calificado",
            "Al imputado se le impuso prisión preventiva por homicidio calificado con ventaja. La defensa sostiene que "
            "tiene arraigo (domicilio, trabajo y familia en Guadalajara) y que el Ministerio Público no justificó el "
            "riesgo de fuga. Necesito resoluciones de apelación sobre medidas cautelares en estos casos.",
            "Prisión preventiva", "penal/homicidio-calificado"),
    Ejemplo("Penal", "Homicidio calificado en grado de tentativa",
            "Mi representado disparó contra otra persona durante una discusión y la lesionó en una pierna; el Ministerio "
            "Público lo acusa de tentativa de homicidio calificado. La defensa sostiene que solo hubo lesiones porque no "
            "se acreditó la intención de matar. ¿Cómo ha resuelto la Sala la reclasificación en apelación?",
            "Sentencia de juicio oral", "penal/homicidio-calificado-en-grado-de-tentativa"),
    Ejemplo("Penal", "Violencia familiar",
            "Mi cliente fue sentenciado por violencia familiar con base casi exclusivamente en la declaración de su "
            "expareja y un dictamen psicológico practicado meses después. Quiero saber si la Sala ha revocado condenas "
            "por insuficiencia probatoria en casos similares.",
            "Sentencia de juicio oral", "penal/violencia-familiar"),
    Ejemplo("Penal", "Abuso sexual infantil",
            "Se condenó a mi defendido por tocamientos a una niña de 9 años. La defensa alega contradicciones entre la "
            "entrevista inicial de la menor y su declaración en juicio, y que la perito no siguió el protocolo. "
            "Busco cómo valora la Sala el testimonio de menores y los agravios de la defensa en apelación.",
            "Sentencia de juicio oral", "penal/abuso-sexual-infantil"),
    Ejemplo("Penal", "Violación",
            "Mi cliente fue vinculado a proceso por violación; la denunciante y él tenían una relación previa y hay "
            "mensajes posteriores al hecho. La defensa apela la vinculación porque el dato de prueba médico no es "
            "concluyente. ¿Qué criterios ha seguido la Sala?",
            "Vinculación a proceso", "penal/violacion"),
    Ejemplo("Penal", "Fraude",
            "El Ministerio Público apeló el auto de no vinculación a proceso dictado a favor de mi cliente, acusado de "
            "fraude por no entregar un terreno que vendió. La defensa sostiene que es un incumplimiento civil de "
            "contrato y no un engaño. Necesito precedentes para contestar los agravios del Ministerio Público.",
            "No vinculación (apela el MP)", "penal/fraude"),
    Ejemplo("Penal", "Despojo de inmuebles",
            "Se vinculó a proceso a mi defendido por despojo porque ocupó una casa que, según él, le cedió verbalmente "
            "el dueño anterior hace años. La defensa sostiene que es un conflicto de posesión que debe resolverse en "
            "la vía civil. Busco resoluciones de la Sala sobre este argumento.",
            "Vinculación a proceso", "penal/despojo-de-inmuebles"),
    Ejemplo("Penal", "Homicidio culposo (tránsito)",
            "Mi cliente atropelló a un peatón que murió; se le condenó por homicidio a título de culpa. La defensa "
            "sostiene que el peatón cruzó fuera de la zona peatonal y que el peritaje de tránsito no determinó la "
            "velocidad. Quiero apelar la individualización de la pena y la reparación del daño.",
            "Sentencia de juicio oral", "penal/homicidio-a-titulo-de-culpa"),
    Ejemplo("Penal", "Narcomenudeo",
            "Detuvieron a mi defendido en la calle y los policías dicen que le encontraron bolsitas con cristal en una "
            "revisión. La defensa alega que la revisión fue ilegal porque no había motivo para detenerlo. Busco "
            "resoluciones de apelación sobre la legalidad de la detención y la exclusión de la prueba.",
            "Vinculación a proceso", "penal/contra-la-salud-en-su-modalidad-de-narcomenudeo"),
    Ejemplo("Penal", "Abandono de familiares",
            "Condenaron a mi cliente por dejar de pagar la pensión de sus hijos durante un año. Él perdió su empleo y "
            "tiene constancias de que buscó trabajo. Busco precedentes en los que la Sala haya valorado la "
            "imposibilidad económica como causa para absolver o modificar la pena.",
            "Sentencia de juicio oral", "penal/abandono-de-familiares"),
    Ejemplo("Penal", "Secuestro agravado",
            "A mi defendido se le acusa de participar en un secuestro como quien cuidaba a la víctima en una casa de "
            "seguridad. La defensa sostiene que no se acreditó su intervención y que el reconocimiento fue "
            "inducido. Necesito precedentes sobre la apelación de sentencias condenatorias por secuestro.",
            "Sentencia de juicio oral", "penal/secuestro-agravado"),
    Ejemplo("Penal", "Lesiones calificadas",
            "Mi cliente lesionó a un vecino con un tubo durante una pelea iniciada por el vecino; se le imputan "
            "lesiones calificadas. La defensa alega legítima defensa. ¿Cómo ha resuelto la Sala la excluyente de "
            "legítima defensa en apelación?",
            "Sentencia de juicio oral", "penal/lesiones-calificadas"),
    Ejemplo("Penal", "Justicia para adolescentes",
            "Mi defendido tiene 16 años y se le impuso una medida de internamiento por robo con violencia. La defensa "
            "considera que la medida es desproporcionada y que debió aplicarse una medida en libertad conforme a la "
            "Ley Nacional del Sistema Integral de Justicia Penal para Adolescentes.",
            "Medida de internamiento", None),
    Ejemplo("Penal", "Ejecución de sentencia",
            "Mi cliente ya cumplió más de la mitad de su pena por robo y el juez de ejecución le negó la libertad "
            "condicionada. Quiero apelar la negativa con precedentes sobre los requisitos de la Ley Nacional de "
            "Ejecución Penal.",
            "Libertad condicionada", None),
]


# La interfaz solo muestra los ejemplos penales (el alcance del proyecto).
EJEMPLOS: list[Ejemplo] = EJEMPLOS_PENALES
