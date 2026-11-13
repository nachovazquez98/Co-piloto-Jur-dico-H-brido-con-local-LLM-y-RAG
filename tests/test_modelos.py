"""Comparación de modelos locales (v11.1, cuarta ronda), sin Ollama ni GPU."""

import json

import pandas as pd

from copiloto_juridico import config, fichas_precalculadas as fp
from copiloto_juridico.evaluacion import modelos as M
from test_fichas_precalculadas import FICHA, TEXTO, LLMFicha, _corpus


def test_misma_muestra_para_todos_los_modelos():
    corpus = pd.DataFrame({"id": [f"d{i:02d}" for i in range(30)], "materia": ["penal"] * 25 + ["civil"] * 5})
    a, b = M.muestra_sentencias(corpus, 10), M.muestra_sentencias(corpus.sample(frac=1, random_state=1), 10)
    assert list(a["id"]) == list(b["id"]) and len(a) == 10 and (a["materia"] == "penal").all()


def test_indicadores_de_fichas_sin_anotacion(tmp_path):
    tabla = fp.generar_todas(_corpus(("a", TEXTO), ("b", TEXTO)), LLMFicha([json.dumps(FICHA), "no es json"]),
                             tmp_path / "f.parquet", log=lambda m: None)
    fila = M.fila_fichas(tabla)
    assert fila["fichas"] == 2 and fila["JSON válido"] == "50.0% (1/2)"
    assert fila["sentido = reglas"] == "100.0% (1/1)"
    assert fila["artículos no encontrados"] == "50.0% (1/2)"  # el 999 no está en la sentencia


def test_omite_los_modelos_no_instalados_sin_descargarlos(monkeypatch):
    descargas = []
    monkeypatch.setattr(M, "instalados", lambda s: {})
    monkeypatch.setattr(M, "descargar_de_memoria", lambda s: descargas.append(s))

    class Store:
        corpus = pd.DataFrame({"id": ["a"], "materia": ["penal"], "texto": [TEXTO]}).set_index("id", drop=False)

    antes = config.FICHAS_PRECALCULADAS
    corrida = M.comparar(Store(), None, ["gemma2:9b", "mistral:7b"], 2, 1, log=lambda m: None)
    assert [a for a in corrida.advertencias if "no está instalado" in a and "no se descarga" in a] and \
        len(corrida.advertencias) == 2
    assert corrida.tablas["Modelos locales"][1].empty
    assert config.FICHAS_PRECALCULADAS == antes  # se restaura al terminar


def test_vram_no_disponible_sin_nvidia_smi(monkeypatch):
    monkeypatch.setattr(M.shutil, "which", lambda _: None)
    with M.MonitorVRAM() as v:
        pass
    assert not v.disponible and v.maximo is None
