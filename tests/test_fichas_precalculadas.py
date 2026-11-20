"""Bloque 2 (v11.1): fichas precalculadas (reanudables, vigentes solo con el mismo texto y versión)."""

import json

import pandas as pd

from copiloto_juridico import config, fichas_precalculadas as fp
from copiloto_juridico.fichas import FichaPenal

FICHA = {"delitos": "Robo calificado", "resumen": "La Sala confirmó el auto.", "sentido_fallo": "CONFIRMA",
         "apelante": "La defensa", "acto_impugnado": "Auto de vinculación a proceso",
         "fundamento_legal": ["artículo 233", "artículo 999"]}
TEXTO = ("vistos para resolver el toca penal 12/2025, formado con motivo del recurso de apelación interpuesto por la "
         "defensa, en contra del auto de vinculación a proceso por robo calificado, art. 233. resolutivos: primero. "
         "se confirma el auto.")


class LLMFicha:
    def __init__(self, respuestas):
        self.settings = config.LLMSettings(provider="openai", model="falso")
        self.respuestas, self.llamadas = list(respuestas), 0

    def generate(self, prompt, json_mode=False):
        self.llamadas += 1
        return self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]


def _corpus(*docs):
    return pd.DataFrame({"id": [d for d, _ in docs], "materia": "penal", "texto": [t for _, t in docs]})


def test_fichas_batch_reanudable_y_vigencia(tmp_path):
    path = tmp_path / "fichas.parquet"
    llm = LLMFicha([json.dumps(FICHA)])
    tabla = fp.generar_todas(_corpus(("a", TEXTO), ("b", TEXTO + " otro")), llm, path, guardar_cada=1, log=lambda m: None)
    assert len(tabla) == 2 and llm.llamadas == 2 and tabla["json_valido"].all()
    fila = tabla.set_index("doc_id").loc["a"]
    assert fila["sentido_reglas"] == "CONFIRMA" and json.loads(fila["articulos_no_encontrados"]) == ["999"]
    fp.generar_todas(_corpus(("a", TEXTO), ("b", TEXTO + " otro")), llm, path, log=lambda m: None)
    assert llm.llamadas == 2  # reanudable: no repite las vigentes
    assert fp.ficha_guardada("a", TEXTO, llm.settings, path).sentido_fallo == "CONFIRMA"
    assert fp.ficha_guardada("a", TEXTO + " cambiado", llm.settings, path) is None  # otro texto
    otro = config.LLMSettings(provider="openai", model="otro")
    assert fp.ficha_guardada("a", TEXTO, otro, path) is None  # otra versión (modelo)


def test_fichas_batch_registra_json_invalido(tmp_path):
    tabla = fp.generar_todas(_corpus(("a", TEXTO)), LLMFicha(["no es json"]), tmp_path / "f.parquet", log=lambda m: None)
    assert not tabla["json_valido"].iloc[0] and tabla["intentos"].iloc[0] == 2


def test_tercer_intento_pide_solo_los_campos_faltantes(tmp_path):
    """v11.2: si la ficha solo carece de 'delitos', se pide aparte y se une; las inválidas se reintentan."""
    sin_delitos = json.dumps({**FICHA, "delitos": "[DATO NO DISPONIBLE]"})
    llm = LLMFicha([sin_delitos, sin_delitos, json.dumps({"delitos": "Robo calificado"})])
    tabla = fp.generar_todas(_corpus(("a", TEXTO)), llm, tmp_path / "f.parquet", log=lambda m: None)
    fila = tabla.iloc[0]
    assert fila["json_valido"] and fila["intentos"] == 3 and llm.llamadas == 3
    assert FichaPenal.desde_guardada(fila["ficha"]).delitos == "Robo calificado"
    # una ficha inválida guardada se vuelve a intentar en la siguiente corrida
    path = tmp_path / "g.parquet"
    fp.generar_todas(_corpus(("a", TEXTO)), LLMFicha(["no es json"]), path, log=lambda m: None)
    llm2 = LLMFicha([json.dumps(FICHA)])
    assert fp.generar_todas(_corpus(("a", TEXTO)), llm2, path, log=lambda m: None)["json_valido"].all()
    assert llm2.llamadas == 1


def test_pipeline_usa_la_ficha_precalculada(monkeypatch):
    from test_pipeline_v11 import ClasificadorFalso, LLMFalso, StoreFalso, _cand, _correr

    ficha = FichaPenal.desde_json(json.dumps(FICHA))
    monkeypatch.setattr(config, "FICHAS_PRECALCULADAS", True)
    monkeypatch.setattr(fp, "ficha_guardada", lambda d, t, s: ficha)
    llm_ = LLMFalso()
    _, res = _correr(StoreFalso([_cand(f"d{i}") for i in range(3)], []), ClasificadorFalso(), llm_, num_precedentes=2)
    assert res.precedentes[0].ficha is ficha
    assert not any("Sentencia:" in p for p in llm_.prompts)  # ninguna llamada de ficha


def test_fichas_batch_sobrevive_a_una_caida_del_servidor(tmp_path, monkeypatch):
    from copiloto_juridico.llm import LLMError

    class LLMCaido(LLMFicha):
        def generate(self, prompt, json_mode=False):
            self.llamadas += 1
            if self.llamadas == 1:
                raise LLMError('POST predict: Post "http://127.0.0.1/completion": EOF')
            return json.dumps(FICHA)

    monkeypatch.setattr(fp.time, "sleep", lambda s: None)
    tabla = fp.generar_todas(_corpus(("a", TEXTO)), LLMCaido([]), tmp_path / "f.parquet", log=lambda m: None)
    assert tabla["json_valido"].all()
