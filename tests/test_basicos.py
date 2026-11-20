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


def test_select_balanced_conserva_las_del_corpus_actual():
    df = pd.DataFrame({"id": [f"d{i:02d}" for i in range(40)], "tipo": "penal/robo",
                       "fecha": [f"2025-01-{i % 28 + 1:02d}" for i in range(40)]})
    actuales = {f"d{i:02d}" for i in range(0, 40, 2)}  # 20 ya en el corpus
    out = select_balanced(df, max_por_tipo=30, conservar=actuales)
    assert len(out) == 30 and actuales <= set(out["id"])  # las nuevas solo llenan el cupo
    assert len(select_balanced(df, max_por_tipo=10, conservar=actuales)) == 10


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


def test_descarga_con_texto_de_otra_sentencia_se_rechaza():
    from copiloto_juridico.datos.corpus import contenido_en, descargas_ajenas

    viejo = "vistos para resolver el toca penal 76/2025 formado con motivo del recurso de apelación " * 8
    completo = viejo.replace("vistos", "vis tos") + " puntos resolutivos: se confirma la resolución apelada."
    otro = "resultando que el juez de control negó la excusa planteada en la carpeta administrativa 12/2024 " * 8
    assert contenido_en(viejo, completo) >= 0.8 and contenido_en(viejo, otro) == 0
    actual = pd.DataFrame({"id": ["a.pdf", "b.pdf"], "texto": [viejo, viejo]})
    nuevos = pd.DataFrame({"id": ["a.pdf", "b.pdf", "c.pdf"], "texto": [completo, otro, otro]})
    assert descargas_ajenas(actual, nuevos)["id"].tolist() == ["b.pdf"]


def test_duplicados_de_texto_conserva_la_fuente_de_consultas():
    from copiloto_juridico.datos.corpus import duplicados_de_texto

    t = "sentencia de la sala penal sobre robo calificado " * 50
    corpus = pd.DataFrame({"id": ["1.pdf", "2.pdf", "3.pdf"], "fecha": ["2025-01-01", "2025-02-01", "2025-03-01"],
                           "texto": [t, t.replace("robo", "r obo"), "otra sentencia distinta " * 50]})
    assert duplicados_de_texto(corpus).to_dict("records") == [{"id": "2.pdf", "igual_a": "1.pdf"}]
    assert duplicados_de_texto(corpus, {"2.pdf"})["id"].tolist() == ["1.pdf"]


def test_extraccion_conserva_inicio_numerado_y_resolutivos():
    """v11.2: "1.VISTOS" numerado marca el inicio y un "Notifíquese" citado de primera instancia no corta la sentencia
    antes de los resolutivos (casos reales de las descargas a mano)."""
    from copiloto_juridico.datos.procesamiento import extract_and_clean_legal_text

    crudo = ("Toca: 318/2025\nPágina:1\n1.VISTOS para resolver el toca penal 318/2025.\n"
             + "el juez de origen resolvió: «QUINTA.- Notifíquese personalmente a las partes». " * 3
             + "consideraciones de la sala sobre los agravios. " * 200
             + "\nANTECEDENTES de la investigación.\n" + "más análisis de los agravios. " * 40
             + "por lo expuesto, esta Sala resuelve con las siguientes proposiciones: primera.- se confirma el auto. "
             + "Notifíquese. Así lo resolvió la Sala. firmas y anexo de datos testados " * 3)
    texto = extract_and_clean_legal_text(crudo)
    assert texto.startswith("1.vistos para resolver")
    assert "se confirma el auto" in texto and "anexo de datos testados" not in texto
