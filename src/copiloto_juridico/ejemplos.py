"""Consultas de ejemplo para la interfaz.

- `EJEMPLOS_PENALES` (v11): casos redactados como los plantearía un defensor que prepara o contesta una
  apelación ante las Salas Penales del STJ, con distintos delitos y actos impugnados. `tipo_corpus` es el tipo
  del corpus que debería encontrarse; si es None, el tema no tiene tipo en el corpus y la app debería avisarlo.
- `EJEMPLOS_V10`: las 18 consultas multimateria del notebook. Se conservan sin cambios porque con ellas se
  midió el "15 de 18" de v10 (Hit@3 de categoría = 0.83, IC 95 % [0.61, 0.94]).

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


EJEMPLOS_V10: list[Ejemplo] = [
    Ejemplo("Penal", "Robo calificado",
            "Se acusa a un individuo de robo calificado por haber sustraído con violencia un teléfono móvil en la vía pública durante la noche. La víctima lo identificó plenamente y existen cámaras de seguridad que registraron el hecho."),
    Ejemplo("Penal", "Lesiones dolosas (riña)",
            "Dos personas riñeron en un bar, resultando una de ellas con lesiones que tardan más de 15 días en sanar. Se imputa el delito de lesiones dolosas al agresor principal, quien argumenta legítima defensa."),
    Ejemplo("Penal", "Homicidio culposo",
            "Un conductor, presuntamente bajo los efectos del alcohol, atropelló a un peatón causándole la muerte. Se le imputa homicidio culposo agravado. La defensa argumenta que el peatón cruzó imprudentemente."),
    Ejemplo("Penal", "Fraude",
            "Se denuncia a una persona por fraude, ya que solicitó un anticipo para realizar una obra de construcción y desapareció sin iniciar los trabajos ni devolver el dinero."),
    Ejemplo("Penal", "Justicia para adolescentes (robo)",
            "Un adolescente es señalado por sustraer mercancía de una tienda departamental. Se busca determinar la medida socioeducativa aplicable conforme a la Ley Nacional del Sistema Integral de Justicia Penal para Adolescentes."),
    Ejemplo("Penal", "Orden de aprehensión (amparo)",
            "Se promueve amparo indirecto contra una orden de aprehensión por considerarla violatoria de derechos fundamentales, argumentando falta de fundamentación y motivación."),
    Ejemplo("Mercantil", "Acción cambiaria directa (pagaré)",
            "Se ejercita acción cambiaria directa por falta de pago de un pagaré vencido. El demandado se niega a pagar alegando que su firma fue falsificada."),
    Ejemplo("Mercantil", "Incumplimiento de contrato de suministro",
            "Una empresa demanda a otra por incumplimiento de contrato de suministro de mercancías, ya que la entrega se realizó fuera de plazo y con productos defectuosos. Se reclaman daños y perjuicios."),
    Ejemplo("Mercantil", "Pago de cheque devuelto",
            "Se demanda el pago de un cheque que fue devuelto por insuficiencia de fondos. El librador argumenta que el cheque fue robado y llenado sin su consentimiento."),
    Ejemplo("Civil", "Rescisión de contrato de arrendamiento",
            "Se demanda la rescisión de un contrato de arrendamiento por falta de pago de rentas por más de tres meses. El inquilino argumenta problemas económicos imprevistos debido a la pérdida de su empleo y solicita una prórroga."),
    Ejemplo("Civil", "Acción reivindicatoria",
            "El propietario de un terreno demanda la acción reivindicatoria contra el poseedor actual, quien construyó una barda invadiendo parte de su propiedad. El demandado alega prescripción adquisitiva."),
    Ejemplo("Civil", "Daños y perjuicios (accidente vehicular)",
            "Se demanda el pago de daños y perjuicios por un accidente vehicular causado por negligencia. El vehículo del demandante sufrió pérdida total. El demandado niega su responsabilidad."),
    Ejemplo("Civil", "Rescisión de compraventa (vicios ocultos)",
            "Un comprador demanda la rescisión de un contrato de compraventa de un inmueble debido a vicios ocultos (humedades estructurales) no informados por el vendedor."),
    Ejemplo("Civil", "Información ad perpetuam / usucapión",
            "Se promueven diligencias de información ad perpetuam para acreditar la posesión de un inmueble no inscrito en el Registro Público, con el fin de obtener un título de propiedad por prescripción."),
    Ejemplo("Familiar", "Divorcio",
            "Se solicita el divorcio necesario por causal de abandono del hogar conyugal por más de seis meses sin causa justificada. Se pide la disolución del vínculo matrimonial y la liquidación de la sociedad conyugal."),
    Ejemplo("Familiar", "Aumento de pensión alimenticia",
            "Se demanda el aumento de la pensión alimenticia para dos menores de edad, argumentando un incremento en las necesidades escolares y médicas, y una mejora en la capacidad económica del deudor alimentario."),
    Ejemplo("Familiar", "Modificación de guarda y custodia",
            "Un progenitor solicita la modificación de la guarda y custodia de un menor, alegando que el otro progenitor expone al niño a un ambiente inadecuado y negligencia en sus cuidados."),
    Ejemplo("Familiar", "Régimen de convivencias",
            "Se demanda la regulación del régimen de convivencias con un menor, ya que el progenitor custodio impide u obstaculiza el contacto del niño con el progenitor no custodio."),
]

EJEMPLOS: list[Ejemplo] = EJEMPLOS_PENALES + [e for e in EJEMPLOS_V10]


def ejemplos_activos(materias: list[str]) -> list[Ejemplo]:
    """Ejemplos de las materias activas (con alcance penal solo se muestran los penales)."""
    activas = {m.lower() for m in materias}
    return [e for e in EJEMPLOS if e.materia.lower() in activas]
