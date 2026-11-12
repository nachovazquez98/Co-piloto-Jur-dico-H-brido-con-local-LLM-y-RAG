"""Regresión de clasificación con la base real (sin LLM: conceptos precalculados en
evaluacion/regresion_conceptos.json). Requiere el modelo de embeddings en la caché local; no requiere GPU."""

import pytest

from copiloto_juridico import config


def _modelo_en_cache() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(config.EMBEDDING_MODEL_NAME, "config.json"), str)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _modelo_en_cache(), reason="modelo de embeddings no descargado")


@pytest.fixture(scope="module")
def store():
    from copiloto_juridico.vectorstore import SentenciasStore

    return SentenciasStore(log=lambda _m: None)


def test_ejemplos_penales_familia_correcta_en_top3(store):
    from copiloto_juridico.clasificador import PesosClasificador, construir_clasificador
    from copiloto_juridico.embeddings import embed_query
    from copiloto_juridico.evaluacion import regresion as rg
    from copiloto_juridico.evaluacion.experimentos import familia_de_categoria
    from copiloto_juridico.evaluacion.familias import cargar_familias

    cache = rg.leer_conceptos()
    familias = cargar_familias(store.corpus["tipo"].unique().tolist())
    clas = construir_clasificador(store.corpus["tipo"].unique().tolist())  # configuración por defecto
    fallos = []
    for e in rg.EJEMPLOS_PENALES:
        if e.sin_tipo_en_corpus:
            continue
        top = rg._top3(store, clas, e.texto, rg.conceptos_para(e.texto, "v11", cache), config.MATERIAS_ACTIVAS,
                       PesosClasificador.desde_config(), embed_query)
        if familias[e.tipo_corpus] not in [familia_de_categoria(t, familias) for t in top]:
            fallos.append((e.etiqueta, top))
    assert not fallos, fallos


def test_consultas_v10_tipo_correcto_en_top3(store):
    from copiloto_juridico.clasificador import ClasificadorHibrido, PesosClasificador
    from copiloto_juridico.embeddings import embed_query
    from copiloto_juridico.evaluacion import regresion as rg

    cache = rg.leer_conceptos()
    clas = ClasificadorHibrido(store.corpus["tipo"].unique().tolist(), "tipo")
    fallos = []
    for c in rg.CASOS_V10:
        top = rg._top3(store, clas, c.ejemplo.texto, rg.conceptos_para(c.ejemplo.texto, "v10", cache),
                       config.MATERIAS, PesosClasificador.desde_config(), embed_query)
        if rg.acierta(top, c.patron_esperado) is False:
            fallos.append((c.ejemplo.tipo, top))
    assert not fallos, fallos
