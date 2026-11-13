"""Resultados guardados para la demo (v11.1, cuarta ronda): clave, ida y vuelta, y que la evaluación no los use."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from copiloto_juridico import cache_resultados as C, config


@pytest.fixture(autouse=True)
def _sin_ollama(monkeypatch):
    monkeypatch.setattr("copiloto_juridico.llm.digest_modelo", lambda s: "sha256:fijo")
    monkeypatch.setattr(config, "CACHE_RESULTADOS", True)


def test_clave_depende_de_lo_que_cambia_el_resultado():
    s = config.LLMSettings()
    base = C.clave("Robo  calificado\n de celular", s)
    assert base == C.clave("robo calificado de celular", s)  # misma consulta normalizada
    assert base != C.clave("robo calificado de un celular", s)
    assert base != C.clave("robo calificado de celular", config.LLMSettings(model="qwen2.5:7b"))
    assert base != C.clave("robo calificado de celular", s, {"num_precedentes": 5})
    assert base == C.clave("robo calificado de celular", s, {"num_precedentes": config.RAG_NUM_RESULTS_DESIRED})


def test_guardar_y_buscar(tmp_path, monkeypatch):
    s = config.LLMSettings()
    res = SimpleNamespace(consulta="robo de celular", sintesis="síntesis")
    assert C.buscar("robo de celular", s, directorio=tmp_path) is None
    C.guardar(res, ["log"], s, directorio=tmp_path)
    g = C.buscar("Robo de celular", s, directorio=tmp_path)
    assert g["resultado"].sintesis == "síntesis" and g["logs"] == ["log"] and g["fecha"]
    monkeypatch.setattr(config, "CACHE_RESULTADOS", False)
    assert C.buscar("robo de celular", s, directorio=tmp_path) is None
    assert C.guardar(res, [], s, directorio=tmp_path) is None


def test_la_evaluacion_nunca_usa_la_cache():
    from copiloto_juridico.evaluacion import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert config.CACHE_RESULTADOS is False
    raiz = Path(C.__file__).parent / "evaluacion"
    assert not [p.name for p in raiz.glob("*.py") if "cache_resultados" in p.read_text(encoding="utf-8")]
