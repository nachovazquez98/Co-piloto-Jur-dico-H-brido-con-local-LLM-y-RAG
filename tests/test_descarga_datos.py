"""descarga_datos/descargar.py sin red: nombres, filtros, tope por tipo, reanudación y reCAPTCHA."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "descarga_datos" / "descargar.py"
spec = importlib.util.spec_from_file_location("descargar", RUTA)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

PDF = b"%PDF-1.4\n...contenido...\n%%EOF\n"


def toca(i, materia, delito, accion=None):
    t = {"id": i, "fecha_publicacion": f"2025-03-20-1151{i:02d}", "materia_data": {"nombre": materia},
         "delito_data": {"nombre": delito}}
    if accion:
        t["accion_especifica_data"] = {"nombre": accion}
    return t


class ClienteFalso:
    """Listado de dos páginas; el PDF del asunto 4 llega truncado."""

    def __init__(self):
        self.paginas = {1: [toca(1, "Penal", "ROBO CALIFICADO"), toca(2, "Penal", "ROBO CALIFICADO"),
                            toca(3, "Civil", "ACCIÓN DE DIVORCIO")],
                        2: [toca(4, "Penal", "FRAUDE"), toca(5, "Familiar", "ALIMENTOS", "INCREMENTO"), {"id": 6}]}
        self.pedidos = []

    def obtener(self, url):
        self.pedidos.append(url)
        if "/tocas?page=" in url:
            n = int(url.split("page=")[1].split("&")[0])
            return 200, json.dumps({"data": {"tocas": {"last_page": 2, "data": self.paginas[n]}}}).encode()
        i = int(url.split("/toca/")[1].split("/")[0])
        return 200, (PDF[:10] if i == 4 else PDF)


def test_nombre_compatible_con_construir_corpus():
    from copiloto_juridico.datos.procesamiento import parse_metadata_from_filename

    a = d.datos_asunto(toca(1, "Penal", "ROBO CALIFICADO"))
    assert a["archivo"] == "2025-03-20-115101_penal_robo-calificado.pdf" and a["tipo"] == "penal/robo-calificado"
    meta = parse_metadata_from_filename(a["archivo"])
    assert meta["materia"] == "penal" and meta["tipo"] == "penal/robo-calificado"
    assert d.datos_asunto({"id": 9}) is None
    assert d.datos_asunto(toca(5, "Familiar", "ALIMENTOS", "INCREMENTO"))["archivo"].endswith("_alimentos_incremento.pdf")


def test_descarga_filtra_respeta_tope_valida_y_reanuda(tmp_path):
    c = ClienteFalso()
    estados = d.descargar(c, tmp_path, materias={"penal"}, max_por_tipo=1, pausa=0)
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["2025-03-20-115101_penal_robo-calificado.pdf"]
    assert estados == {"descargado": 1, "tope alcanzado": 1, "pdf incompleto": 1}
    assert not list(tmp_path.glob("*.part"))
    # Segunda corrida: lo completo no se vuelve a pedir
    c2 = ClienteFalso()
    assert d.descargar(c2, tmp_path, materias={"penal"}, max_por_tipo=1, pausa=0)["ya estaba"] == 1
    assert not any(u.endswith("/toca/1/file") for u in c2.pedidos)
    filas = list(csv.DictReader((tmp_path / "registro_descargas.csv").open(encoding="utf-8")))
    assert len(filas) == 6 and {f["estado"] for f in filas} >= {"descargado", "ya estaba"}


def test_todas_las_materias_filtro_de_delito_y_catalogo(tmp_path):
    estados = d.descargar(ClienteFalso(), tmp_path, delito="divorcio", pausa=0)
    assert estados == {"descargado": 1} and next(tmp_path.glob("*.pdf")).name.endswith("_civil_accion-de-divorcio.pdf")
    cat = tmp_path / "cat"
    assert d.descargar(ClienteFalso(), cat, solo_catalogo=True) == {"catalogado": 5}
    assert not list(cat.glob("*.pdf")) and (cat / "catalogo_asuntos.csv").exists()
    # El tope descuenta lo que ya tiene el corpus
    otra = tmp_path / "otra"
    estados = d.descargar(ClienteFalso(), otra, materias={"penal"}, max_por_tipo=1, pausa=0,
                          ya_en_corpus=d.Counter({"penal/robo-calificado": 1}))
    assert estados["tope alcanzado"] == 2 and not list(otra.glob("*robo*"))


def test_api_con_recaptcha_se_detiene(monkeypatch):
    class Resp:
        status_code, content, headers = 403, b'{"error":"require_recaptcha"}', {}

    cliente = d.ClienteApi()
    monkeypatch.setattr(cliente.sesion, "get", lambda *a, **k: Resp())
    with pytest.raises(d.VerificacionRequerida):
        cliente.obtener(d.API + "/tocas?page=1")
