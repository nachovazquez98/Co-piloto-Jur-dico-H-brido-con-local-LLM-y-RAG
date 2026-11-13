"""Robustez (v11.1, cuarta ronda) con LLM simulado: ninguna consulta difícil rompe el pipeline, la síntesis no
pronostica, el balance lo arma el código y los datos personales no se escriben en disco."""

import builtins
import os
import time

import numpy as np
import pytest

from copiloto_juridico import config, pipeline
from copiloto_juridico.evaluacion import robustez as R
from test_pipeline_v11 import ClasificadorFalso, LLMFalso, StoreFalso, _cand


@pytest.fixture(autouse=True)
def _sin_embeddings(monkeypatch):
    monkeypatch.setattr(pipeline, "embed_query", lambda t: np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(config, "FICHAS_PRECALCULADAS", False)


class LLMObediente(LLMFalso):
    """Sigue la instrucción metida en la consulta (lo que no debe pasar con el LLM real)."""

    def stream(self, prompt):
        yield "La apelación tendrá éxito con seguridad [P1]."


def _eventos(consulta, llm_):
    store = StoreFalso([_cand(f"d{i}") for i in range(6)], [])
    return list(pipeline.Copiloto(store, ClasificadorFalso(), llm_).analizar(consulta, pipeline.Opciones(num_precedentes=2)))


def test_casos_cubren_lo_pedido():
    cats = " ".join(c.categoria for c in R.CASOS)
    for x in ("ortografía", "corta", "larga", "civil", "familiar", "fuera de tema", "datos personales", "instrucciones"):
        assert x in cats
    assert len(next(c for c in R.CASOS if "larga" in c.categoria).consulta) > 3000
    assert next(c for c in R.CASOS if c.privada).consulta.count(R.DATOS_INVENTADOS[1]) == 1


def test_ninguna_consulta_rompe_el_pipeline_ni_escribe_en_disco(monkeypatch):
    escrituras = []
    abrir = builtins.open

    def espia(archivo, modo="r", *a, **kw):
        if any(x in modo for x in "wax+"):
            escrituras.append(str(archivo))
        return abrir(archivo, modo, *a, **kw)

    monkeypatch.setattr(builtins, "open", espia)
    for caso in R.CASOS:
        fila = R.revisar(caso, _eventos(caso.consulta, LLMFalso()))
        assert fila["terminó"] == "sí" and fila["sin pronóstico"] == "sí" and fila["balance por reglas"] == "sí", fila
        if caso.privada:
            assert not any(d in str(fila) for d in R.DATOS_INVENTADOS)
    assert escrituras == []


def test_detecta_pronostico_si_el_modelo_obedece_la_inyeccion():
    caso = next(c for c in R.CASOS if "instrucciones" in c.categoria)
    fila = R.revisar(caso, _eventos(caso.consulta, LLMObediente()))
    assert fila["sin pronóstico"].startswith("no (") and "tendrá éxito" in fila["sin pronóstico"]


def test_busca_los_datos_inventados_solo_en_archivos_nuevos(tmp_path):
    viejo, nuevo = tmp_path / "viejo.log", tmp_path / "sub" / "nuevo.log"
    viejo.write_text(R.DATOS_INVENTADOS[0])
    os.utime(viejo, (0, 0))
    nuevo.parent.mkdir()
    nuevo.write_text(f"consulta: {R.DATOS_INVENTADOS[1]}")
    (tmp_path / "limpio.txt").write_text("nada")
    assert R.archivos_con_datos(tmp_path, time.time() - 60) == ["sub/nuevo.log"]
