"""La interfaz se ejecuta sin errores (Streamlit en modo prueba, sin GPU ni Ollama)."""

from pathlib import Path

import pandas as pd
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[1] / "app"


def test_solo_dos_paginas():
    assert sorted(p.name for p in (APP / "pages").glob("*.py")) == ["1_Como_funciona.py"]


def test_como_funciona_sin_resultados_dice_por_medir(tmp_path, monkeypatch):
    import copiloto_juridico.evaluacion.reporte as rep

    monkeypatch.setattr(rep.ultimo_resultado, "__defaults__", (None, tmp_path))
    at = st_testing.AppTest.from_file(str(APP / "pages" / "1_Como_funciona.py"), default_timeout=60).run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown)
    assert textos.count("Por medir") == 4 and "Agravios" in textos


def test_como_funciona_oculta_diagnosticos(tmp_path, monkeypatch):
    import copiloto_juridico.evaluacion.reporte as rep

    cor = rep.Corrida("recuperacion", borrador=True)
    cor.tabla("Clasificación de la consulta", "d", pd.DataFrame([{"variante": "x", "Hit@3 tipo": 0.5}]))
    cor.tabla("Diagnóstico", "d", pd.DataFrame([{"x": 1}]), diagnostico=True)
    cor.guardar(tmp_path)
    monkeypatch.setattr(rep.ultimo_resultado, "__defaults__", (None, tmp_path))
    at = st_testing.AppTest.from_file(str(APP / "pages" / "1_Como_funciona.py"), default_timeout=60).run()
    assert not at.exception and len(at.dataframe) == 1
    assert any("BORRADOR" in c.value for c in at.caption)


def _modelo_en_cache() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from copiloto_juridico import config

        return isinstance(try_to_load_from_cache(config.EMBEDDING_MODEL_NAME, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _modelo_en_cache(), reason="modelo de embeddings no descargado")
def test_pantalla_principal_con_resultado():
    from copiloto_juridico.fichas import FichaPenal
    from copiloto_juridico.pipeline import AVISO_SIN_MISMO_TIPO, Precedente, Resultado
    from copiloto_juridico.vectorstore import Candidato

    ficha = FichaPenal.desde_json('{"delitos": "Robo calificado", "resumen": "Se confirma.", "sentido_fallo": '
                                  '"CONFIRMA", "acto_impugnado": "Auto de vinculación a proceso", "identificador": '
                                  '"Toca 32/2025", "agravios": [{"agravio": "Reconocimiento ilegal", "calificacion": '
                                  '"INFUNDADO"}], "fundamento_legal": ["artículo 233", "artículo 161"]}')
    c = Candidato("doc.pdf", "penal", "robo-calificado", "penal/robo-calificado", "2025-01-01", 0.86, 0.0321, "x")
    p = Precedente(1, c, "Filtro estricto", ficha.informe_markdown(["161"]), ficha.campos_markdown(), False,
                   ficha=ficha, articulos_no_encontrados=["161"], sentido_reglas="MODIFICA",
                   verificar=["sentido del fallo: la ficha dice CONFIRMA, los resolutivos dicen MODIFICA"])
    tabla = pd.DataFrame({"tipo": ["penal/familia/robo"], "categoría": ["PENAL · familia: robo"], "score_híbrido": [0.4],
                          "coseno": [0.83], "jaccard": [0.1], "kem": [0.0], "voto_recuperación": [1.0],
                          "concisión": [0.7], "tokens": [1], "penalidad": [1.0]})
    res = Resultado("consulta", "penal/familia/robo", "robo calificado", tabla, "Búsqueda amplia (sin filtro)", [p],
                    "La síntesis.", [{"documento": "x", "tipo": "t", "motivo": "QC_1 relevancia"}], 12.0, "llama3.1:8b",
                    0, AVISO_SIN_MISMO_TIPO, 5)
    at = st_testing.AppTest.from_file(str(APP / "Copiloto.py"), default_timeout=240)
    at.session_state["resultado"] = res
    at.run()
    assert not at.exception
    assert [b.label for b in at.button] == ["Buscar precedentes"]
    assert not at.tabs  # sin pestañas: un solo desplegable de detalles técnicos
    assert [e.label for e in at.main.expander] == ["Ficha completa", "Detalles técnicos"]
    assert [e.label for e in at.sidebar.expander] == ["Configuración"]
    assert at.metric[0].label == "Sentido del fallo" and at.metric[0].value == "MODIFICA · verificar"
    textos = " ".join(m.value for m in at.markdown)
    assert "Reconocimiento ilegal" in textos and "no encontrado en la sentencia" in textos
    assert any(AVISO_SIN_MISMO_TIPO in w.value for w in at.warning)
