"""v11.2: evaluación automática sin anotadores humanos (jueces LLM simulados, sin GPU ni Ollama)."""

import json

import pandas as pd
import pytest

from copiloto_juridico import config
from copiloto_juridico.evaluacion import jueces as J, referencias as R
from copiloto_juridico.evaluacion.conjuntos import Consulta, ErrorValidacion, exigir_revisadas
from copiloto_juridico.fichas import FichaPenal
from copiloto_juridico.metadatos import NO_DETERMINADO


class Juez:
    def __init__(self, respuesta):
        self.settings = config.LLMSettings(provider="openai", model="juez-falso")
        self.respuesta, self.llamadas = respuesta, 0

    def generate(self, prompt, json_mode=False):
        self.llamadas += 1
        return self.respuesta(prompt) if callable(self.respuesta) else self.respuesta


def test_normaliza_respuestas_a_valores_cerrados():
    r = J.normalizar("sentencia", 'Aquí va: {"familia": "Robo", "acto_impugnado": "Vinculación", "apelante": '
                     '"la defensa", "sentido": "Se revoca y se ordena REPONER"}', ["robo", "homicidio"])
    assert r == {"familia": "robo", "acto_impugnado": "vinculación", "apelante": "defensa", "sentido": "REPONE"}
    assert J.normalizar("relevancia", '{"relevancia": "2", "mismo_delito": "sí", "mismo_problema_procesal": "no"}') == \
        {"relevancia": 2, "mismo_delito": True, "mismo_problema_procesal": False}
    assert J.normalizar("relevancia", '{"relevancia": 5}') is None
    assert J.normalizar("consulta", "no es json") is None


def test_juez_reanudable_y_rejuzga_si_cambia_el_prompt(tmp_path):
    items = [J.Item("consulta", "A1", "p1"), J.Item("consulta", "A2", "p2")]
    juez = Juez('{"es_consulta": "si", "delito_identificable": "no"}')
    r = J.juzgar(items, juez, directorio=tmp_path, log=lambda m: None)
    assert r["juzgados"] == 2 and juez.llamadas == 2
    J.juzgar(items, juez, directorio=tmp_path, log=lambda m: None)
    assert juez.llamadas == 2  # nada pendiente
    items[1] = J.Item("consulta", "A2", "p2 cambiado")
    J.juzgar(items, juez, directorio=tmp_path, log=lambda m: None)
    assert juez.llamadas == 3
    v = J.vigentes(items, J.leer("juez-falso", tmp_path))
    assert v[("consulta", "A1")] == {"es_consulta": True, "delito_identificable": False}


def test_respuesta_invalida_se_reintenta_una_vez_y_queda_none(tmp_path):
    juez = Juez("sin json")
    r = J.juzgar([J.Item("consulta", "A1", "p")], juez, directorio=tmp_path, log=lambda m: None)
    assert juez.llamadas == 2 and r["invalidas"] == 1
    assert J.vigentes([J.Item("consulta", "A1", "p")], J.leer("juez-falso", tmp_path)) == {("consulta", "A1"): None}


def test_mayoria_absoluta_entre_todos_los_lectores():
    assert R.mayoria(["CONFIRMA", "CONFIRMA", "REVOCA"]) == "CONFIRMA"
    assert R.mayoria(["CONFIRMA", "REVOCA", "MODIFICA"]) == NO_DETERMINADO
    assert R.mayoria(["CONFIRMA", None, "CONFIRMA"]) == "CONFIRMA"
    assert R.mayoria(["CONFIRMA", None, None]) == NO_DETERMINADO  # un voto de tres no es mayoría
    assert R.mayoria(["CONFIRMA", "REVOCA"]) == NO_DETERMINADO


def _sentencias():
    penal = pd.DataFrame({"id": ["d1", "d2", "d3"], "tipo": ["penal/robo-calificado", "penal/homicidio", "penal/robo"]})
    familias = {"penal/robo-calificado": "robo", "penal/robo": "robo", "penal/homicidio": "homicidio"}
    meta = {d: {"acto_impugnado": "vinculación", "apelante": "defensa", "sentido": "CONFIRMA"} for d in penal["id"]}
    bueno = {d: {"familia": f, "acto_impugnado": "vinculación", "apelante": "defensa", "sentido": "REVOCA"}
             for d, f in (("d1", "robo"), ("d2", "homicidio"), ("d3", "robo"))}
    malo = {d: {**r, "familia": "homicidio"} for d, r in bueno.items()}
    return penal, familias, meta, {"j1": bueno, "j2": malo}


def test_valida_jueces_contra_el_stj_y_arma_el_consenso():
    penal, familias, meta, resp = _sentencias()
    tabla = R.tabla_sentencias(penal, familias, meta, resp)
    val = R.validacion_jueces(tabla, ["j1", "j2"])
    assert val.set_index("juez")["se usa"].to_dict() == {"j1": "sí", "j2": "no"}  # j2 acierta 1 de 3
    con = R.con_consenso(tabla, R.usables(val))
    # reglas (CONFIRMA) contra j1 (REVOCA): con dos lectores no hay mayoría
    assert (con["sentido:consenso"] == NO_DETERMINADO).all() and (con["acto_impugnado:consenso"] == "vinculación").all()
    con3 = R.con_consenso(tabla, ["j1", "j2"])
    assert (con3["sentido:consenso"] == "REVOCA").all()
    ac = R.acuerdo_lectores(tabla, ["j1", "j2"])
    assert set(ac["campo"]) == set(R.CAMPOS)


def test_juicios_automaticos_mediana_solo_si_respondieron_todos():
    pares = pd.DataFrame({"consulta_id": ["A1", "A1"], "doc_id": ["d2", "d3"], "consulta": "x"})
    c = Consulta("A1", "mi cliente robó", "d1", "penal/robo-calificado", "robo", "prueba")
    doc_familia = {"d1": "robo", "d2": "homicidio", "d3": "robo"}
    actos = {"d1": "vinculación", "d2": "vinculación", "d3": "sentencia definitiva"}
    resp = {"j1": {"A1|d2": {"relevancia": 0, "mismo_delito": False, "mismo_problema_procesal": True},
                   "A1|d3": {"relevancia": 2, "mismo_delito": True, "mismo_problema_procesal": True}},
            "j2": {"A1|d2": {"relevancia": 1, "mismo_delito": False, "mismo_problema_procesal": True}}}
    t = R.juicios_automaticos(pares, {"A1": c}, doc_familia, actos, resp, ["j1", "j2"])
    reglas = t[t["anotador"] == "reglas"].set_index("doc_id")
    assert reglas.loc["d2", "relevancia"] == 1 and reglas.loc["d3", "relevancia"] == 1
    cons = t[t["anotador"] == "consenso"].set_index("doc_id")
    assert list(cons.index) == ["d2"]  # en d3 falta el juez j2
    assert cons.loc["d2", "relevancia"] == 1 and cons.loc["d2", "mismo_problema_procesal"] == "sí"
    assert len(R.acuerdo_relevancia(t)) == 3


def test_consultas_validadas_por_reglas_y_jueces():
    cs = [Consulta("A1", "mi cliente tomó un celular amenazando con un cuchillo y lo vincularon", "d1", "penal/robo",
                   "robo", "prueba"),
          Consulta("A2", "lo siento, no puedo ayudar con eso", "d1", "penal/robo", "robo", "prueba"),
          Consulta("A3", "represento a una persona a quien el juez le impuso una resolución que considero injusta y quiero apelarla ante la sala", "d1", "penal/robo", "robo", "prueba")]
    resp = {"j": {"A1": {"es_consulta": True, "delito_identificable": True},
                  "A3": {"es_consulta": True, "delito_identificable": False}}}
    v = R.validar_consultas(cs, {"d1": "texto de la sentencia"}, resp, ["j"]).set_index("id")
    assert v.loc["A1", "validada"] and not v.loc["A2", "validada"] and not v.loc["A3", "validada"]
    assert "no se identifica" in v.loc["A3", "motivo"]


def test_prueba_acepta_consultas_validadas_automaticamente():
    c = Consulta("A1", "x", "d1", "t", "f", "prueba", borrador=True)
    with pytest.raises(ErrorValidacion):
        exigir_revisadas([c])
    c.extra["validada"] = True
    exigir_revisadas([c])


def test_fichas_contra_la_referencia():
    ficha = FichaPenal.desde_json(json.dumps({"delitos": "Robo calificado", "resumen": "r", "sentido_fallo": "CONFIRMA",
                                              "apelante": "La defensa", "acto_impugnado": "Auto de vinculación a proceso"}))
    fichas = pd.DataFrame([{"doc_id": "d1", "json_valido": True, "ficha": ficha.a_json(),
                            "articulos_no_encontrados": "[]"}])
    ref = pd.DataFrame([{"doc_id": "d1", "particion": "prueba", "familia_stj": "robo",
                         "acto_impugnado:consenso": "vinculación", "apelante:consenso": "defensa",
                         "sentido:consenso": "REVOCA"}])
    comp = R.comparar_fichas(fichas, ref)
    fila = comp.iloc[0]
    assert fila["familia_ok"] and fila["acto_impugnado_ok"] and fila["apelante_ok"] and not fila["sentido_ok"]
    t = R.tabla_exactitud_fichas(comp).set_index("campo")
    assert t.loc["sentido", "exactitud"] == 0 and t.loc["familia", "exactitud"] == 1


def test_coherencia_de_familias_con_embeddings():
    import numpy as np

    from copiloto_juridico.evaluacion.familias import coherencia

    # tipos a1, a2 (familia A) cerca de (1, 0); b1 (familia B) cerca de (0, 1); x (asignado a B) está junto a A
    docs, emb, doc_tipo = [], [], {}
    for tipo, v in (("a1", (1, 0.1)), ("a2", (1, -0.1)), ("b1", (0.1, 1)), ("b2", (-0.1, 1)), ("x", (1, 0))):
        for i in range(3):
            d = f"{tipo}-{i}"
            docs.append(d), emb.append(v), doc_tipo.update({d: tipo})
    fam = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "x": "B"}
    t = coherencia(docs, np.array(emb), doc_tipo, fam).set_index("tipo")
    assert t.loc["a1", "coincide"] and t.loc["b1", "coincide"]
    assert not t.loc["x", "coincide"] and t.loc["x", "familia más cercana"] == "A"


def test_esfuerzo_simulado_cuenta_sentencias_abiertas():
    from copiloto_juridico.evaluacion.esfuerzo import TOPE, abiertas_hasta, tipo_sentencia

    assert abiertas_hasta(["a", "x", "b", "y", "c"], {"a", "b", "c"}) == 5
    assert abiertas_hasta(["x", "a"], {"a", "b", "c"}) == TOPE + 1
    assert tipo_sentencia("sentencia definitiva") == "Definitiva" and tipo_sentencia("vinculación") == "Interlocutoria"


def test_ampliar_plantilla_no_toca_las_existentes():
    from copiloto_juridico.evaluacion.conjuntos import ampliar_plantilla

    corpus = pd.DataFrame({"id": [f"d{i}" for i in range(14)], "materia": "penal", "texto": "x" * 50,
                           "tipo": ["penal/robo"] * 7 + ["penal/robo-calificado"] * 7})
    fam = {"penal/robo": "robo", "penal/robo-calificado": "robo"}
    existentes = [Consulta("A007", "c", "d0", "penal/robo", "robo", "prueba")]
    nuevas = ampliar_plantilla(corpus, fam, existentes, n_por_familia=4)
    assert [c.id for c in nuevas] == ["A008", "A009", "A010"]
    assert sum(c.particion == "prueba" for c in nuevas) == 1 and "d0" not in {c.doc_id_fuente for c in nuevas}
    assert nuevas[0].tipo == "penal/robo-calificado"  # primero el tipo aún no cubierto


def test_comparacion_por_tarea_puntaje_y_decision():
    from copiloto_juridico.evaluacion.modelos import cache_conceptos, decision, familia_modelo, puntaje_ficha

    assert [familia_modelo(x) for x in ("qwen2.5:7b", "gemma2:9b", "llama3.1:8b", "mistral:7b")] == \
        ["qwen", "gemma", "llama", "mistral"]
    assert cache_conceptos(config.LLMSettings().model) == config.CONCEPTOS_CACHE_PATH
    assert cache_conceptos("qwen2.5:7b").name == "conceptos-qwen2.5-7b.json"
    fila = {"familia_ok": True, "acto_impugnado_ok": False, "apelante_ok": None, "sentido_ok": True}
    assert puntaje_ficha(fila) == pytest.approx(2 / 3) and puntaje_ficha(None) == 0.0  # sin ficha válida = 0
    assert decision({"¿significativa?": "no", "Δ": 0.1}).startswith("sin diferencia")
    assert decision({"¿significativa?": "sí", "Δ": 0.1}) == "mejor que el de la app"
    assert decision({"¿significativa?": "sí", "Δ": -0.1}) == "peor que el de la app"
