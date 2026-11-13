"""Bloque 5 (v11.1): re-ranking opcional con cross-encoder (apagado por defecto)."""

from copiloto_juridico import config, reranker
from copiloto_juridico.vectorstore import Candidato


class ModeloFalso:
    def predict(self, pares, batch_size=8):
        return [len(frag) for _, frag in pares]  # "más largo = más relevante"


def _c(d):
    return Candidato(d, "penal", "robo", "penal/robo", "2025", .8, .1, "")


def test_reranker_apagado_por_defecto():
    assert config.RERANKER is None
    cands = [(_c("a"), "f"), (_c("b"), "f")]
    assert reranker.reordenar("q", cands, lambda d: d) == cands


def test_reranker_reordena_solo_los_primeros(monkeypatch):
    monkeypatch.setattr(reranker, "_modelo", lambda nombre, disp: ModeloFalso())
    textos = {"a": "x", "b": "xxx", "c": "xx", "d": "xxxxxxxx"}
    cands = [(_c(d), "f") for d in "abcd"]
    out = reranker.reordenar("q", cands, textos.get, nombre="falso", max_candidatos=3)
    assert [c.doc_id for c, _ in out] == ["b", "c", "a", "d"]  # "d" queda al final aunque sea el más largo
