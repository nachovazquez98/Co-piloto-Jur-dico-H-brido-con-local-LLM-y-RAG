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
        self.tipos = sorted(set(self.categorias) | set(self.miembros))

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
        self.ficha = {"delitos": "Robo calificado", "resumen": "La Sala confirmó el auto.", "sentido_fallo": "CONFIRMA",
                      "acto_impugnado": "Auto de vinculación a proceso", "fundamento_legal": ["artículo 233"],
                      "agravios": [{"agravio": "Reconocimiento ilegal", "calificacion": "INFUNDADO", "razon": "x"}]}

    def generate(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        if json_mode:
            return json.dumps(self.ficha)
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
    monkeypatch.setattr(config, "FICHAS_PRECALCULADAS", False)  # las pruebas no dependen de data/fichas


def _correr(store, clasif, llm_, **op):
    eventos = list(pipeline.Copiloto(store, clasif, llm_).analizar("robo de celular", pipeline.Opciones(**op)))
    res = [e for e in eventos if e.tipo == "resultado"]
    return eventos, res[0].datos["resultado"] if res else None


def test_veredictos_ambiguos_se_cuentan(monkeypatch):
    store = StoreFalso([_cand(f"d{i}") for i in range(6)], [])
    # Flujo v11 (ficha JSON en una llamada + QC_2 determinista): solo QC_1 puede ser ambiguo
    _, res = _correr(store, ClasificadorFalso(), LLMFalso(qc1="Depende del contexto."), num_precedentes=2)
    assert res.veredictos_ambiguos == 2 and all(p.ambiguo_qc1 for p in res.precedentes)
    assert res.llamadas_llm == 1 + 2 * 2 + 1  # conceptos + (QC_1 + ficha) por precedente + síntesis
    # Flujo de comparación (dos llamadas + QC_2 con LLM)
    monkeypatch.setattr(config, "FICHA_MODO", "dos_llamadas")
    monkeypatch.setattr(config, "QC2_MODO", "llm")
    _, res = _correr(store, ClasificadorFalso(), LLMFalso(qc1="Depende del contexto.", qc2="Puede ser."),
                     num_precedentes=2)
    assert res.veredictos_ambiguos == 4 and all(p.ambiguo_qc1 and p.ambiguo_qc2 for p in res.precedentes)
    assert "Veredictos ambiguos:** 4" in res.markdown()
    assert res.llamadas_llm == 1 + 2 * 4 + 1


def test_ficha_json_una_llamada_y_qc2_determinista():
    store = StoreFalso([_cand(f"d{i}") for i in range(4)], [])
    llm_ = LLMFalso()
    _, res = _correr(store, ClasificadorFalso(), llm_, num_precedentes=1)
    p = res.precedentes[0]
    assert p.ficha.sentido_fallo == "CONFIRMA" and p.ficha.agravios[0]["calificacion"] == "INFUNDADO"
    assert "#### Agravios y su calificación" in p.informe and "QC_2" not in " ".join(llm_.prompts)
    assert p.datos_penales["acto_impugnado"].startswith("Auto de vinculación")
    assert "coherente" in p.veredicto_reporte
    # Delito de otra familia que la consulta y que la etiqueta de la sentencia: QC_2 lo descarta
    llm_ = LLMFalso()
    llm_.ficha = {**llm_.ficha, "delitos": "Fraude genérico"}
    _, res = _correr(StoreFalso([_cand("d0")], []), ClasificadorFalso(), llm_, num_precedentes=1)
    assert res is None or not res.precedentes


def test_ficha_json_reintenta_y_cae_a_dos_llamadas():
    class LLMRoto(LLMFalso):
        def generate(self, prompt, json_mode=False):
            return "no es json" if json_mode else super().generate(prompt)

    llm_ = LLMRoto()
    _, res = _correr(StoreFalso([_cand(f"d{i}") for i in range(3)], []), ClasificadorFalso(), llm_, num_precedentes=1)
    assert res.precedentes[0].ficha is None  # se generó con la extracción en dos llamadas


def test_qc1_contexto_configurable(monkeypatch):
    assert config.QC1_CONTEXTO == "fragmento_similar"  # por defecto en v11
    monkeypatch.setattr(config, "QC1_CONTEXTO", "inicio")
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
    assert prompts.prompts_penal("v10")[0] is prompts.PROMPT_EXTRACCION_PENAL
    assert config.PROMPT_PENAL_VERSION == "v11"  # por defecto v11
    ext, rep = prompts.prompts_penal()
    for campo in ("Acto Impugnado", "Agravios Planteados", "Calificación de los Agravios", "Sentido del Fallo"):
        assert campo in ext
    assert "CONFIRMA, MODIFICA, REVOCA, REPONE u OTRO" in ext and "Agravios y su Calificación" in rep
    monkeypatch.setattr(config, "FICHA_MODO", "dos_llamadas")
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
    c = cl.ClasificadorHibrido(tipos, nivel="familia", familias=familias, enriquecer=False)
    assert c.tipos == ["civil/accion-de-divorcio", "penal/familia/fraude", "penal/familia/robo"]
    assert c.tipos_de("penal/familia/robo") == ["penal/robo-calificado", "penal/robo-simple"]
    q = _emb_falsos(["robo"])[0]
    rec = [SimpleNamespace(tipo="penal/robo-simple", relevancia=0.03), SimpleNamespace(tipo="penal/fraude", relevancia=0.01)]
    res = c.clasificar("me robaron el celular", "robo", q, recuperados=rec)
    assert res.tipo == "penal/familia/robo"
    assert cl.etiqueta_tipo("penal/familia/delitos-sexuales") == "PENAL · familia: delitos sexuales"


def test_familia_sin_palabras_de_relleno(monkeypatch):
    """El embedding de la familia no lleva "y otros" ni nombres compuestos con otro delito; con una sola materia,
    "penal" no cuenta para Jaccard."""
    monkeypatch.setattr(cl, "embed_texts", _emb_falsos)
    tipos = ["penal/armas", "penal/portacion-de-arma-de-fuego-y-otros", "penal/portacion-de-arma-de-fuego-sin-licencia",
             "penal/disparo-de-arma-de-fuego-y-abuso-sexual-infantil"]
    c = cl.ClasificadorHibrido(tipos, nivel="familia", familias={t: "armas" for t in tipos}, enriquecer=True)
    texto = c._texto_embedding("penal/familia/armas", "penal armas")
    assert "otros" not in texto and "abuso" not in texto and "portacion de arma de fuego sin licencia" in texto
    assert c.tokens == [{"arma"}]


def test_lista_de_tokens_desactivada_por_defecto(monkeypatch):
    assert cl.PesosClasificador.desde_config().penal_tokens is False
    assert config.KEM_WEIGHT == 0 and config.JACCARD_WEIGHT == 0.40
    monkeypatch.setattr(config, "USAR_LISTA_TOKENS_PENALIZADOS", True)
    assert cl.PesosClasificador.desde_config().penal_tokens is True


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

    from copiloto_juridico.ejemplos import EJEMPLOS, EJEMPLOS_PENALES, EJEMPLOS_VICTIMA
    from copiloto_juridico.evaluacion.regresion import EJEMPLOS_V10

    tipos = set(pd.concat([pd.read_parquet(f, columns=["tipo"])
                           for f in sorted(glob.glob(str(config.CORPUS_DIR / "sentencias-*.parquet")))])["tipo"])
    assert len(EJEMPLOS_PENALES) >= 12 and len(EJEMPLOS_V10) == 6
    for e in EJEMPLOS_PENALES + EJEMPLOS_VICTIMA:
        if e.sin_tipo_en_corpus:
            assert "sin tipo en el corpus" in e.etiqueta
        else:
            assert e.tipo_corpus in tipos, e.tipo_corpus
    assert {e.etapa for e in EJEMPLOS_PENALES} >= {"Vinculación a proceso", "Prisión preventiva", "Sentencia de juicio oral"}
    assert EJEMPLOS == EJEMPLOS_VICTIMA and all(e.materia == "Penal" for e in EJEMPLOS)
    # Los de la app los redacta el abogado de la víctima: ninguno pide quitarle los cargos a su cliente.
    assert len({e.etiqueta for e in EJEMPLOS}) == len(EJEMPLOS)
    assert all(("mi client" in e.texto.lower() or "represento" in e.texto.lower()) for e in EJEMPLOS)
    assert not any(x in e.texto.lower() for e in EJEMPLOS for x in ("mi defendido", "mi representado", "inocente"))


def test_prompt_conceptos_v11(monkeypatch):
    from copiloto_juridico.evaluacion import conjuntos as cj
    from copiloto_juridico.evaluacion import experimentos as ex

    monkeypatch.setattr(config, "PROMPT_CONCEPTOS_VERSION", "v10")
    assert prompts.prompt_conceptos() is prompts.PROMPT_EXTRACCION_ENTIDADES
    c = cj.Consulta("A1", "texto", None, None, None, "prueba")
    clave_v10 = ex.clave_conceptos(c)
    monkeypatch.setattr(config, "PROMPT_CONCEPTOS_VERSION", "v11")
    assert "PRIMER concepto debe ser SIEMPRE el nombre del delito" in prompts.prompt_conceptos()
    assert ex.clave_conceptos(c) != clave_v10  # la caché no mezcla versiones
    llm_ = LLMFalso()
    _correr(StoreFalso([_cand(f"d{i}") for i in range(4)], []), ClasificadorFalso(), llm_, num_precedentes=1)
    assert "PRIMER concepto" in llm_.prompts[0]


def test_qc1_descarta_todo_muestra_no_verificados():
    store = StoreFalso([_cand(f"d{i}") for i in range(5)], [])
    _, res = _correr(store, ClasificadorFalso(), LLMFalso(qc1="VEREDICTO: NO"), num_precedentes=2)
    assert res is not None and len(res.precedentes) == 2
    assert all(p.no_verificado_qc1 for p in res.precedentes)
    assert not any(d["documento"] in {"d0", "d1"} for d in res.descartados)
    assert "VEREDICTO: SI o NO" in prompts.prompt_qc1("v10") and "MISMO problema procesal" in prompts.prompt_qc1()


def test_verificar_ficha_marca_sentido_y_articulos():
    texto = ("... considerando ... artículo 233 del código penal ... r e s u e l v e primera. - se modifica el auto "
             "de vinculación a proceso de fecha 24 veinticuatro de enero de 2025")
    c = _cand("d1", texto=texto)
    ext = "- **Sentido del Fallo:** CONFIRMA\n- **Fundamento Legal:** artículos 233 y 161 del código penal"
    prec = pipeline.Precedente(1, c, "Filtro estricto", "informe", ext, False)
    pipeline.verificar_ficha(prec)
    assert prec.sentido_reglas == "MODIFICA" and prec.datos_penales["sentido_fallo"] == "MODIFICA"
    assert prec.datos_penales["sentido_llm"] == "CONFIRMA"
    assert prec.articulos_no_encontrados == ["161"] and len(prec.verificar) == 2


def test_delitos_elegidos_por_el_usuario_sustituyen_al_clasificador():
    """v11.2: con delitos elegidos, el filtro estricto usa sus tipos (no los del clasificador) y, si el clasificador
    habría buscado en otra familia, se avisa sin cambiar la búsqueda."""
    miembros = {"penal/familia/robo": ["penal/robo", "penal/robo-calificado"],
                "penal/familia/homicidio": ["penal/homicidio"], "penal/familia/lesiones": ["penal/lesiones"]}
    store = StoreFalso([_cand(f"h{i}", "penal/homicidio") for i in range(6)], [])
    clasif = ClasificadorFalso(["penal/familia/robo"], miembros)
    llm_ = LLMFalso()
    llm_.ficha["delitos"] = "Homicidio calificado"  # QC_2 descarta una ficha de robo si se eligió homicidio
    _, res = _correr(store, clasif, llm_, num_precedentes=1, familias=("homicidio", "lesiones"))
    assert ["penal/homicidio", "penal/lesiones"] in store.llamadas
    assert ["penal/robo", "penal/robo-calificado"] not in store.llamadas
    assert res.familias_elegidas == ("homicidio", "lesiones")
    assert "robo" in res.nota_delito and "Delitos elegidos" in res.markdown()

    store = StoreFalso([_cand(f"r{i}") for i in range(6)], [])
    _, res = _correr(store, clasif, LLMFalso(), num_precedentes=1, familias=("robo",))
    assert res.nota_delito is None  # coincide con el clasificador: sin nota
    _, res = _correr(store, clasif, LLMFalso(), num_precedentes=1)
    assert res.familias_elegidas == () and ["penal/robo", "penal/robo-calificado"] in store.llamadas


def test_sugerencia_inicial_de_delitos_sin_llm():
    miembros = {"penal/familia/robo": ["penal/robo"], "penal/familia/homicidio": ["penal/homicidio"]}
    clasif = ClasificadorFalso(["penal/familia/robo", "penal/familia/homicidio"], miembros)
    assert pipeline.sugerir_familias(StoreFalso([], []), clasif, "me acusan de robo") == ["robo", "homicidio"]
