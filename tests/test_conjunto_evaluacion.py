"""Efecto mínimo detectable y borradores estrictos (v11.1)."""

import pandas as pd
import pytest

from copiloto_juridico.evaluacion import metricas as m
from copiloto_juridico.evaluacion.anotacion import _problemas, palabras_prohibidas
from copiloto_juridico.evaluacion.conjuntos import Consulta


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
