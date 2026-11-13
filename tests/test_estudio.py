"""Estudio de uso (v11.1, cuarta ronda): asignación balanceada, validación del registro y resumen."""

import pandas as pd
import pytest

from copiloto_juridico.evaluacion import estudio as E


def test_asignacion_balanceada_y_orden_alternado():
    a = E.asignacion(4, ["U1", "U2", "U3", "U4", "U5"])
    assert (pd.crosstab(a["caso"], a["metodo"]) == 2).all().all()           # cada caso, 2 y 2
    assert set(pd.crosstab(a["participante"], a["metodo"]).sum(axis=1)) == {5}
    primeros = a.groupby("participante").head(1)
    assert list(primeros["metodo"]) == ["manual", "app", "manual", "app"]  # el método inicial alterna
    assert list(primeros["caso"]) == ["U1", "U5", "U1", "U5"]


def test_casos_desde_los_ejemplos_en_borrador():
    casos = E.casos_desde_ejemplos()
    assert len(casos) == 5 and all(c["borrador"] for c in casos) and len({c["consulta"] for c in casos}) == 5


def _registro(tmp_path, filas):
    p = tmp_path / "registro.csv"
    pd.DataFrame(filas, columns=E.COLUMNAS).to_csv(p, index=False)
    return p


def test_resumen_exploratorio_con_diferencia_pareada(tmp_path):
    filas = []
    for i, part in enumerate(["P01", "P02", "P03", "P04"]):
        for k, caso in enumerate(["U1", "U2", "U3"]):
            metodo = E.METODOS[(k + i) % 2]
            filas.append([part, caso, metodo, 20 if metodo == "manual" else 8 + k, 2 if metodo == "manual" else 3, 4, ""])
    filas.append(["P05", "U1", "app", None, None, None, ""])  # aún sin hacer: se ignora
    df = E.leer_registro(_registro(tmp_path, filas))
    assert len(df) == 12
    corrida = E.resumir(df, [{"id": "U1", "borrador": True}])
    assert corrida.borrador and any("exploratorio" in a for a in corrida.advertencias)
    met = corrida.tablas["Por método"][1].set_index("método")
    assert met.loc["manual", "mediana de minutos"] == 20 and met.loc["app", "mediana de minutos"] == 9
    assert met.loc["manual", "precedentes útiles por minuto"] == pytest.approx(2 / 20)
    dif = corrida.tablas["Diferencia pareada por caso (manual − app, minutos)"][1].iloc[0]
    assert dif["casos"] == 3 and dif["diferencia media"] == pytest.approx(11) and dif["IC excluye 0"] == "sí"


def test_registro_invalido(tmp_path):
    with pytest.raises(ValueError, match="filas inválidas"):
        E.leer_registro(_registro(tmp_path, [["P01", "U1", "a mano", 10, 1, 3, ""]]))
    with pytest.raises(ValueError, match="filas inválidas"):
        E.leer_registro(_registro(tmp_path, [["P01", "U1", "app", 10, 1, 7, ""]]))
