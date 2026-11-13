"""Bloque 4 (v11.1): consultas reales, piloto de anotación, efecto mínimo detectable y borradores estrictos."""

import json

import pandas as pd
import pytest

from copiloto_juridico.evaluacion import metricas as m
from copiloto_juridico.evaluacion.anotacion import _problemas, kappa_piloto, palabras_prohibidas
from copiloto_juridico.evaluacion.conjuntos import Consulta, ErrorValidacion, cargar_todas


def _linea(id_, **kw):
    d = {"id": id_, "consulta": "Mi cliente fue detenido.", "doc_id_fuente": None, "tipo": None, "familia": None,
         "particion": "desarrollo", "sin_respuesta": False, **kw}
    return json.dumps(d, ensure_ascii=False)


def test_consultas_reales_con_esquema_comentado(tmp_path):
    derivadas, reales = tmp_path / "d.jsonl", tmp_path / "r.jsonl"
    derivadas.write_text(_linea("A001") + "\n")
    reales.write_text("# esquema: {...}\n" + _linea("R001") + "\n")
    todas = cargar_todas(derivadas, reales)
    assert [(c.id, c.origen) for c in todas] == [("A001", "derivada"), ("R001", "real")]
    reales.write_text(_linea("A001") + "\n")
    with pytest.raises(ErrorValidacion, match="repetidos"):
        cargar_todas(derivadas, reales)
    assert len(cargar_todas(derivadas, tmp_path / "no_existe.jsonl")) == 1


def test_kappa_piloto_y_desacuerdos():
    hoja = pd.DataFrame({"consulta_id": list("aabbcc"), "doc_id": list("xyxyxy"), "consulta": "q",
                         "relevancia_a": ["2", "0", "1", "1", "0", ""], "relevancia_b": ["2", "0", "2", "1", "0", "1"],
                         "nota": ""})
    resumen, desacuerdos = kappa_piloto(hoja)
    assert resumen["pares"] == 5 and resumen["acuerdo_exacto"] == pytest.approx(0.8)
    assert desacuerdos[["consulta_id", "doc_id"]].values.tolist() == [["b", "x"]]
    assert kappa_piloto(hoja.assign(relevancia_b=""))[0] == {"pares": 0}


def test_efecto_minimo_detectable():
    a, b = [1, 0, 1, 1, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 1, 1]
    mde = m.efecto_minimo_detectable(a, b)
    assert mde == pytest.approx(2.8016 * pd.Series([0, 0, 1, 0, 0, 1, 0, 0]).std() / 8 ** 0.5, rel=1e-3)
    assert m.efecto_minimo_detectable([1, 1], [1, 1]) is None  # sin variación


def test_borrador_estricto_detecta_palabras_del_tipo_y_cifras():
    assert palabras_prohibidas("penal/robo-calificado") == ["robo", "calificado"]
    c = Consulta("A1", "", "d", "penal/robo-calificado", "robo", "desarrollo")
    texto = "la sentencia dice otra cosa completamente distinta"
    c.consulta = "Mi cliente tomó un celular ajeno con amenazas en la calle."
    assert _problemas(c, texto, palabras_prohibidas(c.tipo)) == ""
    c.consulta = "Mi cliente cometió un robo con amenazas."
    assert "robo" in _problemas(c, texto, palabras_prohibidas(c.tipo))
    c.consulta = "Mi cliente tomó un celular el 28 de junio de 2024."
    assert "identifican" in _problemas(c, texto, [])
    c.consulta = "Lo siento, pero no puedo generar contenido relacionado con actividades ilegales."
    assert "se negó" in _problemas(c, texto, [])
