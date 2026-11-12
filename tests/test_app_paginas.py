"""Las páginas nuevas de la app se ejecutan sin errores (Streamlit en modo prueba, sin GPU ni Ollama)."""

import json
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[1] / "app"


def test_pagina_como_funciona():
    at = st_testing.AppTest.from_file(str(APP / "pages" / "2_Como_funciona.py"), default_timeout=60).run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown)
    assert "512 tokens" in str(at.table[0].value.to_dict()) or "512" in textos
    assert "Agravios" in textos and "Precedente vs. jurisprudencia" in textos


def test_pagina_evaluacion_sin_resultados_dice_por_medir(tmp_path, monkeypatch):
    from copiloto_juridico import config

    monkeypatch.setattr(config, "RESULTADOS_DIR", tmp_path)
    import copiloto_juridico.evaluacion.reporte as rep

    monkeypatch.setattr(rep.ultimo_resultado, "__defaults__", (None, tmp_path))
    at = st_testing.AppTest.from_file(str(APP / "pages" / "3_Evaluacion.py"), default_timeout=60).run()
    assert not at.exception
    assert sum("Por medir" in m.value for m in at.markdown) == 6


def test_pagina_evaluacion_muestra_resultados(tmp_path, monkeypatch):
    import pandas as pd

    import copiloto_juridico.evaluacion.reporte as rep

    cor = rep.Corrida("recuperacion", parametros={"particion": "desarrollo"}, borrador=True)
    cor.tabla("Clasificación de la consulta", "desc", pd.DataFrame([{"variante": "x", "Hit@3 tipo": 0.5}]))
    cor.guardar(tmp_path)
    monkeypatch.setattr(rep.ultimo_resultado, "__defaults__", (None, tmp_path))
    at = st_testing.AppTest.from_file(str(APP / "pages" / "3_Evaluacion.py"), default_timeout=60).run()
    assert not at.exception
    assert any("BORRADOR" in e.value for e in at.error)
    assert len(at.dataframe) == 1
