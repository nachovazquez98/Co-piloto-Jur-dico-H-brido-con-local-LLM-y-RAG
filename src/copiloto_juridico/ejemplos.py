"""Consultas de ejemplo (del notebook), etiquetadas por materia y tipo de delito/acción."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ejemplo:
    materia: str
    tipo: str
    texto: str

    @property
    def etiqueta(self) -> str:
        return f"{self.materia} · {self.tipo}"


EJEMPLOS: list[Ejemplo] = [
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
