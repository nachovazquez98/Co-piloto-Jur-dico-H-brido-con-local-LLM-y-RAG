import pandas as pd

from copiloto_juridico.datos.balance import select_balanced, slugify, tipo_key
from copiloto_juridico.datos.procesamiento import pages_to_read, parse_metadata_from_filename
from copiloto_juridico.embeddings import chunk_text
from copiloto_juridico.pipeline import es_fallo, parse_veredicto, recortar_contexto


def test_parse_metadata_from_filename():
    meta = parse_metadata_from_filename("2025-03-14-193036_civil_accion-de-divorcio_otras.pdf")
    assert meta == {"fecha": "2025-03-14", "materia": "civil", "delito_o_accion": "accion-de-divorcio_otras",
                    "tipo": "civil/accion-de-divorcio_otras"}
    assert parse_metadata_from_filename("sin-formato.pdf") is None


def test_tipo_key_normaliza_separadores():
    assert tipo_key("Civil", "prestacion-de-servicios-_otros") == "civil/prestacion-de-servicios_otros"
    assert slugify("ACCIÓN DE PAGO") == "accion-de-pago"


def test_select_balanced_aplica_tope_y_reparte_en_el_tiempo():
    df = pd.DataFrame({"id": [f"d{i}" for i in range(100)], "tipo": ["penal/robo"] * 90 + ["civil/x"] * 10,
                       "fecha": [f"2025-01-{i % 28 + 1:02d}" for i in range(100)]})
    out = select_balanced(df, max_por_tipo=30)
    conteo = out.groupby("tipo").size().to_dict()
    assert conteo == {"penal/robo": 30, "civil/x": 10}
    fechas = sorted(out[out.tipo == "penal/robo"]["fecha"])
    assert fechas[0] == "2025-01-01" and fechas[-1] == "2025-01-28"


def test_pages_to_read_muestrea_pdfs_largos():
    assert pages_to_read(10) == list(range(10))
    paginas = pages_to_read(200)
    assert len(paginas) == 40 and paginas[:10] == list(range(10)) and paginas[-1] == 199


def test_chunk_text_conserva_inicio_y_final():
    texto = "".join(f"{i:05d}" for i in range(20000))
    chunks = chunk_text(texto, size=1000, overlap=100, max_chunks=5)
    assert len(chunks) == 5
    assert texto.startswith(chunks[0]) and texto.endswith(chunks[-1][-50:])


def test_parse_veredicto():
    assert parse_veredicto("1. Contexto...\n2. Contexto...\nVEREDICTO: SI") is True
    assert parse_veredicto("Análisis de la decisión y sin relación.\nVeredicto: NO") is False
    assert parse_veredicto("**Veredicto (SI/NO):** Sí") is True
    assert parse_veredicto("texto sin veredicto claro") is None


def test_es_fallo_y_recorte():
    assert es_fallo("Lo siento, no puedo ayudar con eso porque no tengo datos suficientes.")
    assert not es_fallo("- **Tribunal:** Primera Sala\n- **Decisión:** Se confirma la sentencia apelada.")
    recortado = recortar_contexto("a" * 1000 + "b" * 1000, 500)
    assert recortado.startswith("a") and recortado.endswith("b") and "omitido" in recortado
