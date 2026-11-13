"""Análisis de errores (v11.1, cuarta ronda): taxonomía, selección de casos y lectura de corridas, sin GPU ni LLM."""

import json

import pandas as pd
import pytest

from copiloto_juridico.evaluacion import errores as E


def _d(familia, relevante, qc1=None, verificar=()):
    return E.Devuelto("doc", "1/2025", familia, "vinculación", relevante, qc1, list(verificar))


def _caso(esperada="robo", top3=("robo", "lesiones", "homicidio"), devueltos=(), **kw):
    return E.Caso("c1", "consulta", esperada, [(f, 0.5) for f in top3], list(devueltos), en_corpus=50, **kw)


def test_taxonomia_en_orden_del_pipeline():
    k = 3
    assert E.tipos_presentes(_caso(esperada=None), k) == ["cobertura"]
    assert E.tipos_presentes(_caso(top3=("lesiones", "homicidio", "fraude")), k) == ["clasificacion"]
    bien = [_d("robo", True)] * 3
    assert E.tipos_presentes(_caso(devueltos=bien), k) == []
    orden = [_d("homicidio", False), _d("robo", True), _d("robo", True)]
    assert E.tipos_presentes(_caso(devueltos=orden), k) == ["orden"]
    qc1 = [_d("robo", True, "NO"), _d("robo", True), _d("robo", True)]
    assert E.tipos_presentes(_caso(devueltos=qc1), k) == ["qc1_descarte"]
    ficha = [_d("robo", True, verificar=["artículos no encontrados en la sentencia: 316"])] + bien[:2]
    c = E.clasificar(_caso(devueltos=orden[:1] + ficha[:2]), k)
    assert (c.tipo, c.tambien) == ("orden", ["ficha"])
    # familia con menos sentencias que k: no es un error de orden si están todas
    pocas = _caso(devueltos=[_d("robo", True), _d("lesiones", False), _d("lesiones", False)])
    pocas.en_corpus = 1
    assert E.tipos_presentes(pocas, k) == []


def test_lo_que_mide_la_corrida_manda():
    c = _caso(devueltos=[_d("robo", True)] * 3, marca_corrida="qc1_descarte")
    assert E.clasificar(c, 3).tipo == "qc1_descarte"
    otro = _caso(devueltos=[_d("robo", True)] * 3, marca_corrida="otro")
    assert E.clasificar(otro, 3).tipo == "otro"


def test_seleccion_un_caso_por_tipo_y_tabla_para_diapositiva():
    k = 3
    casos = [("r1", E.clasificar(_caso(devueltos=[_d("homicidio", False), _d("robo", True), _d("robo", True)]), k)),
             ("r1", E.clasificar(_caso(devueltos=[_d("homicidio", False)] * 2 + [_d("robo", True)]), k)),
             ("r1", E.clasificar(_caso(esperada=None, devueltos=[_d("robo", None)] * 3), k))]
    elegidos = E.seleccionar(casos)
    assert [c.tipo for _, c in elegidos] == ["orden", "cobertura"]
    assert sum(bool(d.relevante) for d in elegidos[0][1].devueltos) == 1  # el más claro
    md = E.markdown_presentacion(elegidos, [{"caso": "Prisión preventiva", "antes (orden RRF)": "robo · robo · homicidio",
                                             "después (familia + acto)": "homicidio ×3", "aciertos": "1/3 → 3/3"}])
    assert "| orden |" in md and "| cobertura |" in md and "Prisión preventiva" in md
    assert "solo 1 de los 3 primeros precedentes es de esa familia" in md


def test_lee_la_corrida_de_generacion(tmp_path, monkeypatch):
    corrida = {"comando": "generacion", "commit": "abc", "corpus_fingerprint": "x", "parametros": {},
               "tablas": {"Detalle por sentencia": {"filas": [
                   {"doc_id": "a.pdf", "sentido_ok": False, "sentido_generado": "CONFIRMA", "sentido_anotado": "REVOCA",
                    "delito_ok": True, "etapa_ok": True, "articulos_inexistentes": ""},
                   {"doc_id": "b.pdf", "sentido_ok": True, "delito_ok": True, "etapa_ok": True,
                    "articulos_inexistentes": ""}]}}}
    (tmp_path / "20261113-000000_abc_generacion.json").write_text(json.dumps(corrida))
    monkeypatch.setattr(E.config, "RESULTADOS_DIR", tmp_path)
    monkeypatch.setattr(E.cargar_corrida, "__defaults__", (tmp_path,))

    class Store:
        corpus = pd.DataFrame({"id": ["a.pdf", "b.pdf"], "tipo": ["penal/robo", "penal/robo"],
                               "materia": ["penal", "penal"],
                               "texto": ["toca penal 582025 vistos", "sin número"]}).set_index("id", drop=False)

    nombre, leida, casos, corregidos, _avisos = E.analizar("20261113-000000_abc", Store(), None)
    assert nombre == "20261113-000000_abc_generacion" and not corregidos
    assert [(c.id, c.tipo) for c in casos] == [("a.pdf", "ficha")]
    assert casos[0].devueltos[0].toca == "58/2025"
    assert "sentido: ficha CONFIRMA, anotado REVOCA" in E.markdown_corrida(nombre, leida, casos, [], [])


def test_corrida_inexistente_o_sin_veredictos(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.cargar_corrida("no-existe", tmp_path)
