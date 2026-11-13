"""Bloque 1 (v11.1): congelar lo que se evalúa y usar la partición de prueba una sola vez."""

import pytest

from copiloto_juridico.evaluacion import congelado as cg


@pytest.fixture
def estado(monkeypatch, tmp_path):
    actual = {"codigo": "c1", "corpus": "k1", "indice": "i1", "fichas_precalculadas": "f1", "configuracion": "x1",
              "prompts": {"qc1": "v11 · aa", "sintesis": "v11 · bb"}, "modelo": "llama3.1:8b", "digest_modelo": "d1"}
    monkeypatch.setattr(cg, "estado_actual", lambda: {**actual, "prompts": dict(actual["prompts"])})
    rutas = {"path": tmp_path / "congelado.json", "registro": tmp_path / "prueba_corridas.jsonl"}
    return actual, rutas


def test_prueba_exige_congelado(estado):
    _, rutas = estado
    with pytest.raises(cg.ErrorCongelado, match="congelar"):
        cg.controlar("prueba", "recuperacion", **rutas)
    assert cg.controlar("desarrollo", "recuperacion", **rutas)["advertencias"]  # desarrollo solo avisa


def test_prueba_una_sola_vez_y_forzar_con_motivo(estado):
    _, rutas = estado
    datos = cg.congelar(rutas["path"])
    assert cg.controlar("prueba", "recuperacion", **rutas) == {"congelado": datos["id"], "advertencias": [],
                                                               "motivo_forzar": None}
    with pytest.raises(cg.ErrorCongelado, match="ya se usó"):
        cg.controlar("prueba", "recuperacion", **rutas)
    info = cg.controlar("todas", "recuperacion", "el abogado corrigió 3 juicios", **rutas)
    assert info["motivo_forzar"] == "el abogado corrigió 3 juicios"
    assert cg.controlar("prueba", "qc", **rutas)["motivo_forzar"] is None  # otro comando: primera vez


def test_cambio_bloquea_prueba_y_avisa_en_desarrollo(estado):
    actual, rutas = estado
    cg.congelar(rutas["path"])
    actual["corpus"] = "k2"
    actual["prompts"]["sintesis"] = "v11 · cc"
    with pytest.raises(cg.ErrorCongelado, match=r"corpus, prompt sintesis"):
        cg.controlar("prueba", "recuperacion", **rutas)
    adv = cg.controlar("desarrollo", "recuperacion", **rutas)["advertencias"]
    assert adv == ["Distinto de lo congelado: corpus, prompt sintesis."]


def test_digest_solo_con_ollama():
    from copiloto_juridico.config import LLMSettings
    from copiloto_juridico.llm import digest_modelo

    assert digest_modelo(LLMSettings(provider="openai")) is None
