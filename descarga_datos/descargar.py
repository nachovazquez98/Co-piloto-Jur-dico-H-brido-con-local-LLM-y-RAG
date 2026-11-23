"""Descarga de sentencias del portal público del Supremo Tribunal de Justicia del Estado de Jalisco (todas las materias).

Reúne y ordena las dos celdas de descarga del notebook original del proyecto (fase 0, commit 8d88a36): la consulta
directa a la API pública y la sesión de navegador con verificación humana. Es independiente de la aplicación: solo
requiere `requests` y, para el modo navegador, `playwright` (`pip install playwright && playwright install chromium`).

Modos:
  api        Consulta directa a la API pública. Funcionó hasta octubre de 2026; si la API exige reCAPTCHA, el script se
             detiene con un aviso (use el modo navegador).
  navegador  Abre una ventana visible de Chromium en el portal. Una persona resuelve el reCAPTCHA y el script continúa
             con esa sesión; si el portal vuelve a pedir la verificación, espera a que la persona la resuelva otra vez.
             El script nunca resuelve ni evita el reCAPTCHA.

Uso (desde la raíz del repositorio):
  python descarga_datos/descargar.py --destino data/raw/nuevas --materias penal --max-por-tipo 30
  python descarga_datos/descargar.py --destino ~/sentencias --materias todas --modo navegador
  python descarga_datos/descargar.py --destino ~/sentencias --materias civil,familiar --delito divorcio
  python descarga_datos/descargar.py --destino ~/sentencias --solo-catalogo     # lista de asuntos, sin PDF

Cada PDF se nombra `fecha-hora_materia_delito[_accion].pdf` (el formato que lee `construir-corpus --pdfs`), se valida
(cabecera %PDF y marcador %%EOF) y se escribe de forma atómica; una descarga interrumpida se reanuda sin repetir lo que
ya está completo. `registro_descargas.csv`, en la carpeta de destino, deja constancia de cada asunto.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

ORIGEN = "https://publicacionsentencias.stjjalisco.gob.mx"
PORTAL = f"{ORIGEN}/sentencias"
API = "https://publica-sentencias-backend.stjjalisco.gob.mx"
FILTROS_VACIOS = ("&sala_id=&numero=&periodo=&derivado=&delito_id=&materia_id=&nombre_delito="
                  "&accion_especifica_id=&accion_simultanea_id=")
MATERIAS = ("penal", "civil", "familiar", "mercantil")  # las del catálogo del portal (data/catalogo/)
REINTENTOS, ESPERA_429 = 5, 30


class VerificacionRequerida(RuntimeError):
    """El portal pidió reCAPTCHA y el modo en uso no permite que una persona lo resuelva."""


# --------------------------------------------------------------------------------------------------------------
# Nombres de archivo y validación (sin red)
# --------------------------------------------------------------------------------------------------------------


def slugify(valor: str) -> str:
    """El mismo slug que usa el paquete (`datos.balance.slugify`): ASCII, minúsculas, guiones."""
    valor = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode("ascii")
    valor = re.sub(r"[^\w\s-]", "", valor.lower())
    return re.sub(r"[-\s]+", "-", valor).strip("-_")


def datos_asunto(toca: dict) -> dict | None:
    """Metadatos de un asunto del listado de la API; None si le falta la fecha, la materia o el delito."""
    fecha = toca.get("fecha_publicacion")
    materia = (toca.get("materia_data") or {}).get("nombre")
    delito = (toca.get("delito_data") or {}).get("nombre")
    if not (toca.get("id") and fecha and materia and delito):
        return None
    acciones = [(toca.get(k) or {}).get("nombre") for k in ("accion_especifica_data", "accion_simultanea_data")]
    partes = [fecha, materia, delito] + [a for a in acciones if a]
    return {"toca_id": toca["id"], "fecha": fecha, "materia": slugify(materia), "delito": delito,
            "tipo": f"{slugify(materia)}/{slugify(delito)}", "archivo": slugify("_".join(partes)) + ".pdf"}


def pdf_completo(contenido: bytes) -> bool:
    return contenido[:5] == b"%PDF-" and b"%%EOF" in contenido[-2048:]


def archivo_completo(ruta: Path) -> bool:
    try:
        with ruta.open("rb") as f:
            cabeza = f.read(5)
            f.seek(max(0, ruta.stat().st_size - 2048))
            return cabeza == b"%PDF-" and b"%%EOF" in f.read()
    except OSError:
        return False


def conteo_por_tipo(destino: Path) -> Counter:
    """Sentencias completas que ya hay en la carpeta, por tipo (materia/delito), para respetar el tope."""
    conteo = Counter()
    for ruta in destino.glob("*.pdf"):
        partes = ruta.stem.split("_")
        if len(partes) >= 3 and archivo_completo(ruta):
            conteo[f"{partes[1]}/{partes[2]}"] += 1
    return conteo


def seleccionado(asunto: dict, materias: set[str] | None, delito: str | None) -> bool:
    if materias is not None and asunto["materia"] not in materias:
        return False
    return delito is None or slugify(delito) in slugify(asunto["delito"])


# --------------------------------------------------------------------------------------------------------------
# Clientes: API directa y sesión de navegador con verificación humana
# --------------------------------------------------------------------------------------------------------------


class ClienteApi:
    """Consulta directa a la API pública (sin verificación). Se detiene si la API exige reCAPTCHA."""

    def __init__(self):
        import requests

        self.requests = requests
        self.sesion = requests.Session()
        self.sesion.headers["User-Agent"] = "copiloto-juridico/11.2 (proyecto academico)"

    def obtener(self, url: str) -> tuple[int, bytes]:
        for intento in range(REINTENTOS):
            try:
                r = self.sesion.get(url, timeout=60)
            except self.requests.RequestException as e:
                print(f"   Error de red ({e}); reintento en {5 * 2 ** intento} s", flush=True)
                time.sleep(5 * 2 ** intento)
                continue
            if r.status_code == 429:
                time.sleep(int(r.headers.get("retry-after", ESPERA_429)))
                continue
            if r.status_code in (401, 403) and b"recaptcha" in r.content.lower():
                raise VerificacionRequerida("La API exige verificación reCAPTCHA: use --modo navegador, en el que una "
                                            "persona la resuelve en la ventana del portal.")
            return r.status_code, r.content
        raise RuntimeError(f"Sin respuesta de {url} tras {REINTENTOS} intentos")

    def cerrar(self) -> None:
        self.sesion.close()


class ClienteNavegador:
    """Sesión del portal en una ventana visible de Chromium. Una persona resuelve el reCAPTCHA; el script detecta que
    el portal cargó el listado y continúa con esa sesión. Si el portal vuelve a pedir la verificación, recarga la
    página y espera a la persona."""

    def __init__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self.navegador = self._pw.chromium.launch(headless=False)
        self.contexto = self.navegador.new_context()
        self.pagina = self.contexto.new_page()
        self.cabeceras = {"Accept": "application/json, text/plain, */*", "Origin": ORIGEN, "Referer": f"{ORIGEN}/"}
        self.verificada = False
        self.pagina.on("request", self._copiar_cabeceras)
        self.pagina.on("response", self._marcar_verificada)
        self.pagina.goto(PORTAL, wait_until="domcontentloaded", timeout=60_000)
        self._esperar_persona("En la ventana de Chromium, resuelva el reCAPTCHA (o pulse «Buscar») hasta que aparezca "
                              "la tabla de sentencias.")

    def _copiar_cabeceras(self, request) -> None:
        if request.url.startswith(API) and "/tocas" in request.url:
            self.cabeceras.update({k: v for k, v in request.headers.items()
                                   if k.lower() not in ("host", "content-length", "connection")})

    def _marcar_verificada(self, response) -> None:
        if response.url.startswith(API) and "/tocas" in response.url and response.status == 200:
            self.verificada = True

    def _esperar_persona(self, mensaje: str) -> None:
        print(f"-> {mensaje}", flush=True)
        while not self.verificada:
            self.pagina.wait_for_timeout(500)
        print("   Sesión verificada; continúa la descarga.", flush=True)

    def obtener(self, url: str) -> tuple[int, bytes]:
        for _ in range(REINTENTOS):
            r = self.contexto.request.get(url, headers=self.cabeceras, timeout=60_000)
            cuerpo = r.body()
            if r.status in (401, 403) and b"recaptcha" in cuerpo.lower():
                self.verificada = False
                self.pagina.reload(wait_until="domcontentloaded")
                self._esperar_persona("El portal pidió de nuevo la verificación: resuélvala en la ventana de Chromium.")
                continue
            if r.status == 429:
                print(f"   El portal pidió esperar (429); pausa de {ESPERA_429} s", flush=True)
                time.sleep(ESPERA_429)
                continue
            return r.status, cuerpo
        raise RuntimeError(f"Sin respuesta de {url} tras {REINTENTOS} intentos")

    def cerrar(self) -> None:
        self.navegador.close()
        self._pw.stop()


# --------------------------------------------------------------------------------------------------------------
# Recorrido del listado
# --------------------------------------------------------------------------------------------------------------


def _pagina(cliente, numero: int) -> dict:
    import json

    estado, cuerpo = cliente.obtener(f"{API}/tocas?page={numero}{FILTROS_VACIOS}")
    if estado != 200:
        raise RuntimeError(f"La página {numero} del listado respondió {estado}")
    return json.loads(cuerpo)["data"]["tocas"]


def conteo_corpus() -> Counter:
    """Sentencias por tipo que ya tiene el corpus del proyecto (requiere el paquete instalado: `pip install -e .`)."""
    from copiloto_juridico.datos.corpus import load_corpus

    return Counter(load_corpus()["tipo"])


def descargar(cliente, destino: Path, materias: set[str] | None = None, delito: str | None = None,
              max_por_tipo: int | None = None, pausa: float = 1.5, paginas: int | None = None,
              solo_catalogo: bool = False, ya_en_corpus: Counter | None = None) -> Counter:
    """Recorre el listado del portal página por página y descarga los asuntos seleccionados. Devuelve el conteo por
    estado (descargado, ya estaba, tope alcanzado, PDF incompleto, error). `ya_en_corpus` se suma al conteo de la
    carpeta para el tope por tipo."""
    destino.mkdir(parents=True, exist_ok=True)
    nombre_registro = "catalogo_asuntos.csv" if solo_catalogo else "registro_descargas.csv"
    registro = destino / nombre_registro
    nuevo = not registro.exists()
    conteo = Counter() if solo_catalogo else conteo_por_tipo(destino) + (ya_en_corpus or Counter())
    estados = Counter()
    ultima = _pagina(cliente, 1)["last_page"]
    ultima = min(ultima, paginas) if paginas else ultima
    with registro.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fecha_registro", "toca_id", "fecha", "materia", "delito", "tipo", "archivo",
                                          "estado"])
        if nuevo:
            w.writeheader()
        for numero in range(1, ultima + 1):
            print(f"--- Página {numero} de {ultima} ---", flush=True)
            for toca in _pagina(cliente, numero)["data"]:
                asunto = datos_asunto(toca)
                if asunto is None or not seleccionado(asunto, materias, delito):
                    continue
                if solo_catalogo:
                    estado = "catalogado"
                else:
                    estado = _descargar_uno(cliente, destino, asunto, conteo, max_por_tipo)
                    if estado == "descargado":
                        time.sleep(pausa)
                estados[estado] += 1
                w.writerow({"fecha_registro": datetime.now().isoformat(timespec="seconds"), **asunto, "estado": estado})
            f.flush()
    return estados


def _descargar_uno(cliente, destino: Path, asunto: dict, conteo: Counter, max_por_tipo: int | None) -> str:
    ruta = destino / asunto["archivo"]
    if ruta.exists() and archivo_completo(ruta):
        return "ya estaba"
    if max_por_tipo is not None and conteo[asunto["tipo"]] >= max_por_tipo:
        return "tope alcanzado"
    try:
        estado, cuerpo = cliente.obtener(f"{API}/toca/{asunto['toca_id']}/file")
    except (RuntimeError, OSError) as e:
        print(f"   Error en el asunto {asunto['toca_id']}: {e}", flush=True)
        return "error"
    if estado != 200:
        print(f"   El asunto {asunto['toca_id']} respondió {estado}", flush=True)
        return "error"
    if not pdf_completo(cuerpo):
        return "pdf incompleto"
    temporal = ruta.with_suffix(".part")
    temporal.write_bytes(cuerpo)
    temporal.replace(ruta)
    conteo[asunto["tipo"]] += 1
    print(f"   Descargado: {ruta.name}", flush=True)
    return "descargado"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Descarga sentencias del portal del STJ Jalisco (todas las materias).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Uso", 1)[1])
    p.add_argument("--destino", required=True, type=Path, help="Carpeta donde se guardan los PDF")
    p.add_argument("--materias", default="todas",
                   help=f"Lista separada por comas ({', '.join(MATERIAS)}) o «todas» (por defecto)")
    p.add_argument("--delito", help="Solo los asuntos cuyo delito o acción contenga este texto")
    p.add_argument("--max-por-tipo", type=int, help="Tope de sentencias por tipo (materia + delito); el corpus usa 30")
    p.add_argument("--modo", choices=["api", "navegador"], default="api")
    p.add_argument("--pausa", type=float, default=1.5, help="Segundos entre descargas (cortesía con el servidor)")
    p.add_argument("--paginas", type=int, help="Recorrer solo las primeras N páginas del listado")
    p.add_argument("--solo-catalogo", action="store_true", help="Solo la lista de asuntos (catalogo_asuntos.csv)")
    p.add_argument("--descontar-corpus", action="store_true",
                   help="El tope por tipo cuenta también las sentencias que ya tiene el corpus del proyecto")
    a = p.parse_args(argv)
    materias = None if a.materias.strip().lower() == "todas" else {slugify(x) for x in a.materias.split(",") if x.strip()}
    cliente = ClienteNavegador() if a.modo == "navegador" else ClienteApi()
    try:
        estados = descargar(cliente, a.destino.expanduser(), materias, a.delito, a.max_por_tipo, a.pausa, a.paginas,
                            a.solo_catalogo, conteo_corpus() if a.descontar_corpus else None)
    except VerificacionRequerida as e:
        sys.exit(f"❌ {e}")
    finally:
        cliente.cerrar()
    print("Resumen: " + ", ".join(f"{k} {v}" for k, v in estados.most_common()), flush=True)


if __name__ == "__main__":
    main()
