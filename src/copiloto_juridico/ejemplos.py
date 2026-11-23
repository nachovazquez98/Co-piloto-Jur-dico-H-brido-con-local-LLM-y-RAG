"""Consultas de ejemplo para la interfaz.

- `EJEMPLOS_VICTIMA` (los que muestra la app): casos redactados por el abogado, cuyo cliente fue la víctima del
  delito (o su familia), que quiere saber cómo han resuelto las Salas Penales del STJ asuntos parecidos. `tipo_corpus` es el tipo del corpus
  que debería encontrarse; si es None, el tema no tiene tipo en el corpus y la app debería avisarlo.
- `EJEMPLOS_PERSONA` y `EJEMPLOS_TRIBUNAL`: los mismos temas planteados por la persona justiciable (sin formación
  jurídica) y por el tribunal de alzada que estudia un toca (perfiles de usuario de la interfaz, `PERFILES`).
- `EJEMPLOS_PENALES` (v11): los mismos temas redactados como los plantearía un defensor en apelación. Ya no se
  muestran en la app; se conservan sin cambios porque la evaluación (`evaluacion/regresion.py`, errores, modelos)
  y sus corridas guardadas se hicieron con ellos.
- Las consultas penales de v10 están en `evaluacion/regresion.py`.

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
    perfil: str = "abogado"           # quién consulta: una de las claves de PERFILES

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


EJEMPLOS_VICTIMA: list[Ejemplo] = [
    Ejemplo("Penal", "Robo calificado",
            "Mi cliente fue asaltado en una parada de camión: un hombre lo amenazó con un cuchillo y le quitó el "
            "celular. Lo reconoció en unas fotografías y el agresor fue vinculado a proceso, pero su defensa apeló el "
            "auto de vinculación. Quiero saber cómo ha resuelto la Sala apelaciones parecidas y qué valor le da al "
            "reconocimiento.",
            "Vinculación a proceso", "penal/robo-calificado"),
    Ejemplo("Penal", "Robo equiparado",
            "A mi cliente le robaron la camioneta y la policía encontró varias de sus autopartes en un taller. "
            "Condenaron al dueño del taller, pero apeló diciendo que las compró sin saber que eran robadas. Quiero "
            "saber cómo ha resuelto la Sala estos casos y qué pasa con la reparación del daño.",
            "Sentencia de juicio oral", "penal/robo-equiparado"),
    Ejemplo("Penal", "Homicidio calificado",
            "Represento a la familia de un hombre asesinado. Al responsable le impusieron prisión preventiva por "
            "homicidio calificado y su defensa apeló la medida para que siga el proceso en libertad. Quiero saber "
            "cómo ha resuelto la Sala las apelaciones sobre prisión preventiva en homicidios.",
            "Prisión preventiva", "penal/homicidio-calificado"),
    Ejemplo("Penal", "Homicidio calificado en grado de tentativa",
            "Un vecino le disparó a mi cliente durante una discusión y lo hirió en una pierna. El Ministerio Público "
            "lo acusa de tentativa de homicidio calificado, pero su defensa pide que se reclasifique como lesiones. "
            "Quiero saber cómo ha resuelto la Sala la reclasificación en apelación.",
            "Sentencia de juicio oral", "penal/homicidio-calificado-en-grado-de-tentativa"),
    Ejemplo("Penal", "Violencia familiar",
            "Mi clienta denunció a su expareja por violencia familiar y lo condenaron con su declaración y un "
            "dictamen psicológico. Él apeló diciendo que las pruebas no son suficientes. Quiero saber cómo valora "
            "la Sala estas pruebas y si suele confirmar las condenas.",
            "Sentencia de juicio oral", "penal/violencia-familiar"),
    Ejemplo("Penal", "Abuso sexual infantil",
            "Represento a la madre de una niña de 9 años que sufrió tocamientos de un familiar. Hubo sentencia "
            "condenatoria, pero la defensa apeló alegando contradicciones entre la entrevista inicial de la niña y "
            "su declaración en juicio. Quiero saber cómo valora la Sala el testimonio de menores en apelación.",
            "Sentencia de juicio oral", "penal/abuso-sexual-infantil"),
    Ejemplo("Penal", "Violación",
            "Mi clienta denunció a su expareja por violación y lo vincularon a proceso. Su defensa apeló porque "
            "tenían una relación previa y el examen médico no fue concluyente. Quiero saber qué criterios ha "
            "seguido la Sala en estos casos.",
            "Vinculación a proceso", "penal/violacion"),
    Ejemplo("Penal", "Fraude",
            "Mi cliente pagó completo un terreno y el vendedor nunca se lo entregó. El juez no vinculó a proceso al "
            "vendedor porque consideró que es un asunto civil, y el Ministerio Público apeló. Quiero saber cómo ha "
            "resuelto la Sala si estos casos son fraude o incumplimiento de contrato.",
            "No vinculación (apela el MP)", "penal/fraude"),
    Ejemplo("Penal", "Despojo de inmuebles",
            "Una persona se metió a vivir en la casa de mi cliente y dice que el dueño anterior se la cedió de "
            "palabra. La vincularon a proceso por despojo, pero su defensa apeló diciendo que es un conflicto civil. "
            "Quiero saber cómo ha resuelto la Sala este argumento.",
            "Vinculación a proceso", "penal/despojo-de-inmuebles"),
    Ejemplo("Penal", "Homicidio culposo (tránsito)",
            "Represento a los hijos de un peatón que murió atropellado. Condenaron al conductor por homicidio "
            "culposo, pero apeló la pena y la reparación del daño diciendo que el peatón cruzó fuera de la zona "
            "peatonal. Quiero saber cómo resuelve la Sala la reparación del daño en estos casos.",
            "Sentencia de juicio oral", "penal/homicidio-a-titulo-de-culpa"),
    Ejemplo("Penal", "Amenazas",
            "Un vecino amenazó varias veces a mi cliente con matarlo, en persona y por mensajes. Ya presentamos la "
            "denuncia por amenazas y quiero saber cómo ha resuelto la Sala estos casos y qué pruebas toma en cuenta.",
            "Sentencia de juicio oral", "penal/amenazas"),
    Ejemplo("Penal", "Abandono de familiares",
            "La expareja de mi clienta dejó de pagar la pensión de sus hijos durante un año. Lo condenaron por "
            "abandono de familiares y apeló diciendo que perdió su empleo. Quiero saber cómo ha resuelto la Sala "
            "estos casos y qué pasa con el pago de lo que debe.",
            "Sentencia de juicio oral", "penal/abandono-de-familiares"),
    Ejemplo("Penal", "Secuestro agravado",
            "Mi cliente fue secuestrado y lo tuvieron varios días en una casa de seguridad. Condenaron a uno de los "
            "que lo cuidaban, pero apeló diciendo que el reconocimiento fue inducido. Quiero saber cómo ha resuelto "
            "la Sala las apelaciones en casos de secuestro.",
            "Sentencia de juicio oral", "penal/secuestro-agravado"),
    Ejemplo("Penal", "Lesiones calificadas",
            "Un vecino golpeó a mi cliente con un tubo durante una discusión y le dejó lesiones graves. Lo acusan de "
            "lesiones calificadas, pero él dice que actuó en legítima defensa. Quiero saber cómo ha resuelto la Sala "
            "la legítima defensa en apelación.",
            "Sentencia de juicio oral", "penal/lesiones-calificadas"),
    Ejemplo("Penal", "Justicia para adolescentes",
            "Un adolescente de 16 años asaltó con violencia a mi cliente y le impusieron una medida de "
            "internamiento. Su defensa apeló porque la considera desproporcionada. Quiero saber cómo se resuelven "
            "estas apelaciones conforme a la Ley Nacional del Sistema Integral de Justicia Penal para Adolescentes.",
            "Medida de internamiento", None),
    Ejemplo("Penal", "Ejecución de sentencia",
            "La persona que robó en el negocio de mi cliente ya cumplió más de la mitad de su pena y pidió la "
            "libertad condicionada; el juez de ejecución se la negó y apeló. Quiero saber qué requisitos de la Ley "
            "Nacional de Ejecución Penal revisa la Sala en estos casos.",
            "Libertad condicionada", None),
]


# Otros perfiles de usuario (v11.2): la persona justiciable (víctima, ofendido o familiar, sin formación jurídica) y
# el tribunal de alzada que estudia un toca.
EJEMPLOS_PERSONA: list[Ejemplo] = [
    Ejemplo("Penal", "Robo calificado",
            "A mi hijo lo asaltaron con una pistola y le quitaron la moto. Ya detuvieron a uno de los asaltantes y nos "
            "dijeron que su abogado apeló. ¿Qué ha decidido el tribunal en casos parecidos?",
            None, "penal/robo-calificado", "persona"),
    Ejemplo("Penal", "Violencia familiar",
            "Mi hermana denunció a su pareja porque la golpeaba y la insultaba. Lo condenaron, pero él apeló. ¿Qué ha "
            "pasado en otros casos parecidos cuando el agresor apela?",
            None, "penal/violencia-familiar", "persona"),
    Ejemplo("Penal", "Fraude",
            "Le compré un terreno a un señor, le pagué todo y nunca me lo entregó. El juez dijo que no era delito y el "
            "Ministerio Público apeló. ¿Cómo ha resuelto el tribunal casos así?",
            None, "penal/fraude", "persona"),
    Ejemplo("Penal", "Homicidio culposo (tránsito)",
            "Mi papá murió atropellado. Al conductor lo condenaron, pero apeló porque no quiere pagar la reparación "
            "del daño que le fijaron. ¿Qué ha decidido el tribunal en casos parecidos?",
            None, "penal/homicidio-a-titulo-de-culpa", "persona"),
]

EJEMPLOS_TRIBUNAL: list[Ejemplo] = [
    Ejemplo("Penal", "Robo calificado",
            "Toca en estudio: la defensa apela el auto de vinculación a proceso por robo calificado; el único dato "
            "de prueba es un reconocimiento por fotografía hecho sin defensor. Busco cómo han calificado las Salas "
            "agravios semejantes, para resolver con un criterio congruente.",
            "Vinculación a proceso", "penal/robo-calificado", "tribunal"),
    Ejemplo("Penal", "Homicidio calificado",
            "Toca en estudio: la defensa apela la prisión preventiva impuesta por homicidio calificado; alega arraigo "
            "y que no se justificó el riesgo de fuga. Busco resoluciones previas de las Salas sobre esta medida "
            "cautelar.",
            "Prisión preventiva", "penal/homicidio-calificado", "tribunal"),
    Ejemplo("Penal", "Abuso sexual infantil",
            "Toca en estudio: la defensa apela la sentencia condenatoria por abuso sexual infantil y alega "
            "contradicciones en el testimonio de la menor. Busco cómo han valorado las Salas el testimonio de "
            "menores en apelación.",
            "Sentencia de juicio oral", "penal/abuso-sexual-infantil", "tribunal"),
    Ejemplo("Penal", "Fraude",
            "Toca en estudio: el Ministerio Público apela el auto de no vinculación a proceso por fraude en la venta "
            "de un terreno; el juez consideró que es un incumplimiento civil. Busco cómo han resuelto las Salas esa "
            "distinción.",
            "No vinculación (apela el MP)", "penal/fraude", "tribunal"),
]

# Perfiles de usuario de la interfaz. Los tres resuelven el mismo problema (localizar y leer los precedentes de las
# Salas Penales); el abogado es el usuario principal.
PERFILES: dict[str, str] = {
    "abogado": "Abogado (defensor o asesor jurídico)",
    "persona": "Persona justiciable",
    "tribunal": "Tribunal de alzada (Sala Penal)",
}

# La interfaz muestra los ejemplos de los tres perfiles, cada uno con los suyos (penal, el alcance del proyecto).
EJEMPLOS: list[Ejemplo] = EJEMPLOS_VICTIMA + EJEMPLOS_PERSONA + EJEMPLOS_TRIBUNAL


def ejemplos_de(perfil: str) -> list[Ejemplo]:
    return [e for e in EJEMPLOS if e.perfil == perfil]
