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


def test_pages_to_read_muestrea_pdfs_largos(monkeypatch):
    from copiloto_juridico import config

    assert pages_to_read(200) == list(range(200))  # v11: se leen completos
    monkeypatch.setattr(config, "PDF_MAX_PAGES_FULL_READ", 50)  # comportamiento de v10
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


ANEXO_REAL = (" no.50 eliminadoelnombrecompleto en 1 renglones por ser un dato identificativo de conformidad con el "
              "lineamiento quincuagésimo octavo fracción i de los lgppicr. no.43 eliminadoelnombrecompleto en 1 renglones "
              "po r ser un dato identificativo de conformid ad con el lineamiento quincuagésimo octavo fracción i de los "
              "lgppicr.")


def test_limpiar_texto_quita_anexo_y_ruido():
    from copiloto_juridico.datos.procesamiento import limpiar_texto

    texto = ("r e s u e l v e primera. se confirma el auto dictado a no.207eliminadoelnombrecompleto1 . 15 --- "
             "tribunal de justicia del estado de jalisco notifíquese." + ANEXO_REAL)
    limpio = limpiar_texto(texto)
    assert "lgppicr" not in limpio and "renglones" not in limpio
    assert limpio.startswith("resuelve primera. se confirma el auto dictado a [eliminado]")
    assert "tribunal de justicia" not in limpio and "---" not in limpio
    assert limpiar_texto(limpio) == limpio  # idempotente


def test_limpiar_texto_encabezados_con_palabras_partidas():
    """Fragmentos reales del corpus: la nota de testado con palabras partidas y el número de expediente que el
    encabezado repite en cada hoja (se conserva la primera aparición)."""
    from copiloto_juridico.datos.procesamiento import limpiar_texto

    hoja = "34 --- toca 202025 el documento fue testado con el programa elid a eliminador de datos judicial del "
    texto = ("toca 202025 apelación contra la sentencia. " + hoja + "primer agravio fundado. " + hoja
             + "segundo agravio infundado. " + hoja + "v i s t o s")
    limpio = limpiar_texto(texto)
    assert limpio == "toca 202025 apelación contra la sentencia. primer agravio fundado. segundo agravio infundado. vistos"
    assert limpiar_texto(limpio) == limpio
    # etiquetas en femenino, plural y con espacios metidos por la extracción
    etiquetas = ("la víctima no.12eliminadalaedad3 , con domicilio no.62eli minadoeldomicilio2 y "
                 "no.14eliminadalainformacióncorrespondienteaunapersonarelacion adaconunprocedimientopenal79 declaró")
    assert limpiar_texto(etiquetas) == "la víctima [eliminado] , con domicilio [eliminado] y [eliminado] declaró"


def test_limpiar_texto_restos_de_encabezado_y_colas_repetidas():
    """Fragmentos reales del corpus (v11.1): resto del encabezado del tribunal intercalado con el expediente,
    entradas del anexo cortadas por páginas omitidas y el encabezado de la sala repetido al final."""
    from copiloto_juridico.datos.procesamiento import limpiar_texto

    assert limpiar_texto("oral electrónico 3592025 de justicia en el estado de jalisco por su nombre") == \
        "oral electrónico 3592025 por su nombre"
    assert limpiar_texto("la procuraduría general de justicia del estado de jalisco informó") == \
        "la procuraduría general de justicia del estado de jalisco informó"
    assert limpiar_texto("fin. ... 6 págs omitidas ... exp. [eliminado] en 1 renglones por ser un dato identificativo "
                         "de conformidad con el lineamiento quincuagésimo ... 2 págs omitidas ... sigue") == \
        "fin. ... 8 págs omitidas ... sigue"
    cola = " ".join(f"décima prime ra sala en mate ria penal {n}" for n in range(80, 90))
    assert limpiar_texto("se confirma. " + cola) == "se confirma. décima prime ra sala en mate ria penal"
    victimas = "las víctimas [eliminado] , [eliminado] , [eliminado] declararon"
    assert limpiar_texto(victimas) == victimas  # las listas de datos testados no son encabezados


def test_recortar_contexto_conserva_agravios():
    relleno = "texto sin interés. " * 2000
    agravios = "el primer agravio es infundado; el segundo agravio es fundado. " * 20
    texto = "INICIO " + relleno + agravios + relleno + " RESOLUTIVOS"
    r = recortar_contexto(texto, 6000)
    assert len(r) <= 6000 + 10 and r.startswith("INICIO") and r.endswith("RESOLUTIVOS")
    assert r.count("agravio es infundado") >= 10
    sin = recortar_contexto("a" * 5000 + "b" * 5000, 4000)  # sin agravios: 55 % / 45 % como en v10
    assert sin.startswith("a") and sin.endswith("b")


def test_pdf_completo_reemplaza_la_version_muestreada():
    """Bloque 7: al reintegrar los PDF vueltos a descargar, gana la versión sin páginas omitidas."""
    from copiloto_juridico.datos.corpus import unir_fuentes

    actual = pd.DataFrame({"id": ["a.pdf", "b.pdf"], "texto": ["inicio ... 12 págs omitidas ... fin", "completo b"],
                           "origen": "chroma"})
    pdfs = pd.DataFrame({"id": ["a.pdf", "b.pdf"], "texto": ["inicio medio fin", "otra versión de b"], "origen": "pdf"})
    unido = unir_fuentes([actual, pdfs]).set_index("id")
    assert unido.loc["a.pdf", "origen"] == "pdf"     # la completa reemplaza a la muestreada
    assert unido.loc["b.pdf", "origen"] == "chroma"  # entre completas, la primera fuente
