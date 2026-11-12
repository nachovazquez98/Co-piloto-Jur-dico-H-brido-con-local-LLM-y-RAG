"""Cambios de v11 en el pipeline, el clasificador y el LLM, con dobles de prueba (sin GPU, sin Ollama)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from copiloto_juridico import clasificador as cl
from copiloto_juridico import config, llm, pipeline, prompts
from copiloto_juridico.clasificador import ResultadoClasificacion
from copiloto_juridico.vectorstore import Candidato


def _cand(doc_id, tipo="penal/robo-calificado", texto="INICIO " + "x" * 3000):
    return Candidato(doc_id, tipo.split("/")[0], tipo.split("/")[1], tipo, "2025-01-01", 0.83, 0.03, texto)


class StoreFalso:
    def __init__(self, estrictos, amplios):
        self.estrictos, self.amplios, self.llamadas = estrictos, amplios, []

    def buscar(self, consulta, tipos=None, query_embedding=None, consulta_lexica=None, **kw):
        self.llamadas.append(tipos)
        return self.estrictos if tipos else self.amplios

    def fragmento_mas_similar(self, doc_id, q, largo=2500):
        return "FRAGMENTO SIMILAR"


class ClasificadorFalso:
    def __init__(self, categorias=("penal/robo-calificado",), miembros=None):
        self.categorias, self.miembros = list(categorias), miembros or {}

    def tipos_de(self, c):
        return self.miembros.get(c, [c])

    def clasificar(self, consulta, entidades, q, recuperados=None):
        tabla = pd.DataFrame({"tipo": self.categorias, "categoría": self.categorias,
                              "score_híbrido": [0.5] * len(self.categorias), "coseno": [0.8] * len(self.categorias),
                              "jaccard": [0.1] * len(self.categorias), "kem": [0.2] * len(self.categorias),
                              "voto_recuperación": [1.0] * len(self.categorias), "penalidad": [1.0] * len(self.categorias)})
        return ResultadoClasificacion(self.categorias[0], tabla, ["robo"])


class LLMFalso:
    def __init__(self, qc1="1. ...\n2. ...\nVEREDICTO: SI", qc2="VEREDICTO: SI"):
        self.settings = SimpleNamespace(model="falso", max_context_chars=10_000)
        self.qc1, self.qc2, self.prompts = qc1, qc2, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if "Verificación de Relevancia" in prompt:
            return self.qc1
        if "verificador de CALIDAD" in prompt:
            return self.qc2
        if "conceptos jurídicos clave" in prompt:
            return "robo, violencia"
        return "- **Tribunal:** Primera Sala Penal\n- **Sentido del Fallo:** CONFIRMA\n- **Decisión Específica:** Se confirma"

    def stream(self, prompt):
        yield "Síntesis de prueba suficientemente larga para no parecer un fallo del modelo."


@pytest.fixture(autouse=True)
def _sin_embeddings(monkeypatch):
    monkeypatch.setattr(pipeline, "embed_query", lambda t: np.zeros(4, dtype=np.float32))


def _correr(store, clasif, llm_, **op):
    eventos = list(pipeline.Copiloto(store, clasif, llm_).analizar("robo de celular", pipeline.Opciones(**op)))
    res = [e for e in eventos if e.tipo == "resultado"]
    return eventos, res[0].datos["resultado"] if res else None


def test_veredictos_ambiguos_se_cuentan():
    store = StoreFalso([_cand(f"d{i}") for i in range(6)], [])
    _, res = _correr(store, ClasificadorFalso(), LLMFalso(qc1="Depende del contexto.", qc2="Puede ser."), num_precedentes=2)
    assert res.veredictos_ambiguos == 4  # 2 precedentes × (QC_1 + QC_2)
    assert all(p.ambiguo_qc1 and p.ambiguo_qc2 for p in res.precedentes)
    assert "Veredictos ambiguos:** 4" in res.markdown()
    assert res.llamadas_llm == 1 + 2 * 4 + 1


def test_qc1_contexto_configurable(monkeypatch):
    store = StoreFalso([_cand(f"d{i}") for i in range(4)], [])
    llm_ = LLMFalso()
    _correr(store, ClasificadorFalso(), llm_, num_precedentes=1)
    assert "INICIO" in next(p for p in llm_.prompts if "Verificación de Relevancia" in p)
    monkeypatch.setattr(config, "QC1_CONTEXTO", "fragmento_similar")
    llm_ = LLMFalso()
    _correr(store, ClasificadorFalso(), llm_, num_precedentes=1)
    assert "FRAGMENTO SIMILAR" in next(p for p in llm_.prompts if "Verificación de Relevancia" in p)


def test_aviso_sin_precedentes_del_mismo_tipo():
    store = StoreFalso([], [_cand("a1", "penal/fraude"), _cand("a2", "penal/fraude")])
    _, res = _correr(store, ClasificadorFalso(), LLMFalso(), num_precedentes=2)
    assert res.aviso_cobertura == pipeline.AVISO_SIN_MISMO_TIPO
    assert pipeline.aviso_cobertura(["Filtro estricto"], {"kem_max": 0.0}) is None  # sin umbral: por medir
    assert "no coincide bien" not in (pipeline.aviso_cobertura(["Filtro estricto"], {"kem_max": 0.5}, 0.2) or "")
    assert pipeline.aviso_cobertura(["Filtro estricto"], {"kem_max": 0.1}, umbral=0.2, senal="kem_max")


def test_filtro_expande_familias_a_tipos():
    store = StoreFalso([_cand(f"d{i}") for i in range(6)], [])
    clasif = ClasificadorFalso(["penal/familia/robo"], {"penal/familia/robo": ["penal/robo", "penal/robo-calificado"]})
    _correr(store, clasif, LLMFalso(), num_precedentes=1)
    assert ["penal/robo", "penal/robo-calificado"] in store.llamadas


def test_prompt_penal_v11_y_datos_de_la_ficha(monkeypatch):
    assert prompts.prompts_penal()[0] is prompts.PROMPT_EXTRACCION_PENAL  # por defecto v10
    monkeypatch.setattr(config, "PROMPT_PENAL_VERSION", "v11")
    ext, rep = prompts.prompts_para("penal")
    for campo in ("Acto Impugnado", "Agravios Planteados", "Calificación de los Agravios", "Sentido del Fallo"):
        assert campo in ext
    assert "CONFIRMA, MODIFICA, REVOCA, REPONE u OTRO" in ext and "Agravios y su Calificación" in rep
    store = StoreFalso([_cand(f"d{i}") for i in range(4)], [])
    llm_ = LLMFalso()
    _, res = _correr(store, ClasificadorFalso(), llm_, num_precedentes=1)
    assert any("Acto Impugnado" in p for p in llm_.prompts)
    assert res.precedentes[0].datos_penales["sentido_fallo"] == "CONFIRMA"
    with pytest.raises(ValueError):
        prompts.prompts_penal("v99")


class _Respuesta:
    def __init__(self, lineas):
        self.lineas = lineas

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        return iter(self.lineas)


def test_seed_se_envia_al_llm(monkeypatch):
    enviados = []

    def post(url, json=None, **kw):
        enviados.append(json)
        if "chat/completions" in url:
            return _Respuesta(['data: {"choices": [{"delta": {"content": "hola"}}]}', "data: [DONE]"])
        return _Respuesta([b'{"message": {"content": "hola"}, "done": true}'])

    monkeypatch.setattr(llm.requests, "post", post)
    s = config.LLMSettings(provider="ollama", seed=7)
    assert llm.get_client(s).generate("x") == "hola"
    assert enviados[-1]["options"]["seed"] == 7
    assert config.LLMSettings().seed == 42
    llm.get_client(config.LLMSettings(provider="openai", seed=9)).generate("x")
    assert enviados[-1]["seed"] == 9


def _emb_falsos(textos, kind="query"):
    rng = np.random.default_rng(0)
    v = np.array([rng.standard_normal(8) + (5 if "robo" in t else 0) for t in textos], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_clasificador_a_nivel_familia(monkeypatch):
    monkeypatch.setattr(cl, "embed_texts", _emb_falsos)
    tipos = ["penal/robo-calificado", "penal/robo-simple", "penal/fraude", "civil/accion-de-divorcio"]
    familias = {"penal/robo-calificado": "robo", "penal/robo-simple": "robo", "penal/fraude": "fraude"}
    c = cl.ClasificadorHibrido(tipos, nivel="familia", familias=familias)
    assert c.tipos == ["civil/accion-de-divorcio", "penal/familia/fraude", "penal/familia/robo"]
    assert c.tipos_de("penal/familia/robo") == ["penal/robo-calificado", "penal/robo-simple"]
    q = _emb_falsos(["robo"])[0]
    rec = [SimpleNamespace(tipo="penal/robo-simple", relevancia=0.03), SimpleNamespace(tipo="penal/fraude", relevancia=0.01)]
    res = c.clasificar("me robaron el celular", "robo", q, recuperados=rec)
    assert res.tipo == "penal/familia/robo"
    assert cl.etiqueta_tipo("penal/familia/delitos-sexuales") == "PENAL · familia: delitos sexuales"


def test_kem_vocabulario_y_lista_de_tokens(monkeypatch, tmp_path):
    vocab = tmp_path / "v.txt"
    vocab.write_text("# comentario\nrobo\ncalificado\n")
    monkeypatch.setattr(config, "VOCABULARIO_KEM_PATH", vocab)
    conceptos = cl.lemmatized_tokens("robo de celular a un menor")
    assert cl.terminos_importantes(conceptos, "conceptos") == conceptos
    # Solo sobrevive el término del vocabulario. Ojo: spaCy lematiza "robo" aislado como "robar"; el
    # vocabulario guarda ambas formas para que coincida con el lema en contexto ("robo").
    assert cl.terminos_importantes(conceptos, "vocabulario") == {"robo"}  # "celular" y "menor" no son clave
    with pytest.raises(ValueError):
        cl.terminos_importantes(conceptos, "otro")
    monkeypatch.setattr(config, "USAR_LISTA_TOKENS_PENALIZADOS", False)
    assert cl.PesosClasificador.desde_config().penal_tokens is False


def test_construir_clasificador_respeta_materias(monkeypatch):
    monkeypatch.setattr(cl, "embed_texts", _emb_falsos)
    monkeypatch.setattr(config, "MATERIAS_ACTIVAS", ["penal"])
    c = cl.construir_clasificador(["penal/robo", "civil/divorcio"], nivel="tipo")
    assert c.tipos == ["penal/robo"]


def test_materias_filtro_del_store():
    from copiloto_juridico.vectorstore import SentenciasStore

    s = SentenciasStore.__new__(SentenciasStore)
    s.doc_materias = np.array(["penal", "civil", "penal"])
    assert s.materias_filtro(["penal", "civil"]) is None  # todas: comportamiento v10
    assert s.materias_filtro(["penal"]) == ["penal"]
    with pytest.raises(ValueError, match="desconocidas"):
        s.materias_filtro(["penal", "pnal"])


def test_ejemplos_penales_existen_en_el_corpus():
    import glob

    from copiloto_juridico.ejemplos import EJEMPLOS_PENALES, EJEMPLOS_V10, ejemplos_activos

    tipos = set(pd.concat([pd.read_parquet(f, columns=["tipo"])
                           for f in sorted(glob.glob(str(config.CORPUS_DIR / "*.parquet")))])["tipo"])
    assert len(EJEMPLOS_PENALES) >= 12 and len(EJEMPLOS_V10) == 18
    for e in EJEMPLOS_PENALES:
        if e.sin_tipo_en_corpus:
            assert "sin tipo en el corpus" in e.etiqueta
        else:
            assert e.tipo_corpus in tipos, e.tipo_corpus
    assert {e.etapa for e in EJEMPLOS_PENALES} >= {"Vinculación a proceso", "Prisión preventiva", "Sentencia de juicio oral"}
    assert all(e.materia == "Penal" for e in ejemplos_activos(["penal"]))
