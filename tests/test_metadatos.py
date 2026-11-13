"""Bloque 3 (v11.1): metadatos procesales por reglas y orden de candidatos por facetas."""

from copiloto_juridico import metadatos as md
from copiloto_juridico.pipeline import ordenar_por_facetas
from copiloto_juridico.vectorstore import Candidato


def test_metadatos_por_reglas_textos_reales():
    """Inicios reales del corpus (con las palabras partidas por la extracción del PDF)."""
    t1 = ("vistos para resolver los autos del toca penal número 3132014 , formado con motivo del recurso de apelación, "
          "interpuesto por el asesor jurídico de la parte agraviada , respecto del auto de no vinculación a proceso , "
          "dictado en la audiencia inicial")
    t2 = ("visto para resolver el toca de apelación, formado con motivo del recurso de apelación interpuest o por l a "
          "agente del ministerio público , en contra de la resolución en la cual se calificó de ilegal la detención")
    t3 = ("vistos para resolver los autos del toca de juicio oral número 2002023, que se formó con motivo del recurso de "
          "apelación directa interpuesto por el agente del ministerio público y el asesor jurídico particular de la "
          "víctima, así como el de apelación adhesiva interpuesto por el imputado y su defensor, en contra de la "
          "sentencia definitiva")
    t4 = ("vistos para resolver el motivo de la apelación promovida por el sentenciado y su defensor de oficio , en co "
          "ntra de la sentencia definitiva condena toria de fecha 10 diez de septiembre")
    assert md.metadatos_de_sentencia(t1)["acto_impugnado"] == "no vinculación"
    assert md.metadatos_de_sentencia(t1)["apelante"] == "asesor de la víctima"
    assert md.metadatos_de_sentencia(t2) == {"acto_impugnado": "otro", "apelante": "ministerio público",
                                             "sentido": md.NO_DETERMINADO}
    assert md.metadatos_de_sentencia(t3)["apelante"] == "varios"
    assert md.metadatos_de_sentencia(t4)["acto_impugnado"] == "sentencia definitiva"
    assert md.metadatos_de_sentencia("texto sin el párrafo inicial")["acto_impugnado"] == md.NO_DETERMINADO
    assert md.acto_de_consulta("A mi cliente lo vincularon a proceso por homicidio") == "vinculación"
    assert md.acto_de_consulta("Le impusieron prisión preventiva oficiosa") == "medida cautelar"
    assert md.acto_de_consulta("Robo de celular en la vía pública") is None


def _c(doc_id, tipo):
    return Candidato(doc_id, "penal", tipo, f"penal/{tipo}", "2025", .8, .1, "")


def test_orden_por_facetas():
    cands = [(_c("r1", "robo"), "f"), (_c("h1", "homicidio"), "f"), (_c("r2", "robo"), "f"), (_c("h2", "homicidio"), "f")]
    familias = {"penal/robo": "robo", "penal/homicidio": "homicidio"}
    actos = {"r1": "medida cautelar", "h1": "vinculación", "r2": "vinculación", "h2": "medida cautelar"}
    orden = ordenar_por_facetas(cands, familias, {"homicidio"}, "medida cautelar", actos)
    assert [c.doc_id for c, _ in orden] == ["h2", "h1", "r1", "r2"]
    sin_acto = ordenar_por_facetas(cands, familias, {"homicidio"}, None, actos)
    assert [c.doc_id for c, _ in sin_acto] == ["h1", "h2", "r1", "r2"]  # estable dentro de cada nivel
