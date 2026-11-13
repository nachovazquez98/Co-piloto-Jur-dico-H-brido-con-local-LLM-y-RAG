"""Bloque 6 (v11.1): la síntesis cita cada afirmación y el código verifica las citas."""

import json

from copiloto_juridico.fichas import FichaPenal
from copiloto_juridico.pipeline import Precedente, verificar_citas
from copiloto_juridico.vectorstore import Candidato

FICHA = {"delitos": "Robo calificado", "resumen": "La Sala confirmó el auto.", "sentido_fallo": "CONFIRMA",
         "apelante": "La defensa"}


def _c(doc_id, tipo):
    return Candidato(doc_id, "penal", tipo, f"penal/{tipo}", "2025", .8, .1, "")


def _prec(n, sentido, delito):
    f = FichaPenal.desde_json(json.dumps({**FICHA, "sentido_fallo": sentido, "delitos": delito}))
    return Precedente(n, _c(f"d{n}", "robo-calificado"), "Filtro estricto", "", f.campos_markdown(), False, ficha=f)


def test_verificar_citas_de_la_sintesis():
    ps = [_prec(1, "CONFIRMA", "Robo calificado"), _prec(2, "REVOCA", "Homicidio calificado")]
    texto = ("### Respuesta breve\n"
             "La Sala confirmó la vinculación por robo porque el reconocimiento fue válido [P1]. "
             "En el otro caso revocó el auto de homicidio por falta de datos de prueba [P2].\n"
             "- La Sala revocó la medida en el caso del robo con violencia [P1].\n"
             "- Se analizó la cadena de custodia de los indicios recabados [P3].\n"
             "Esta frase no cita ningún precedente aunque es bastante larga.")
    marcado, conteo = verificar_citas(texto, ps)
    assert conteo == {"afirmaciones": 5, "con_cita_valida": 2, "sin_cita": 1, "marcadas": 2}
    assert "[P1]. _(verificar: la ficha citada dice CONFIRMA)_" in marcado
    assert "_(verificar: no hay precedente 3)_" in marcado and marcado.startswith("### Respuesta breve\n")


def test_una_vineta_es_una_afirmacion():
    """Caso real: la viñeta trae el agravio y la razón en dos oraciones y la cita va al final."""
    ps = [_prec(1, "CONFIRMA", "Homicidio calificado")]
    texto = ("* El imputado argumentó que la prisión preventiva no era necesaria. — **INFUNDADO**: Porque no se "
             "evidenció un cambio objetivo en las condiciones que justificaron la medida [P1].")
    marcado, conteo = verificar_citas(texto, ps)
    assert conteo == {"afirmaciones": 1, "con_cita_valida": 1, "sin_cita": 0, "marcadas": 0} and marcado == texto
