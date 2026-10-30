"""Fase 0: descarga balanceada de sentencias desde la API pública del STJ Jalisco.

Recorre el listado de tocas y descarga una sentencia solo si su tipo (materia + delito/acción) aún no
alcanza su meta (`min(30, disponibles)` según el catálogo). Termina cuando todos los tipos del catálogo
están completos o se agota el listado. Cada PDF se valida (cabecera, %%EOF, páginas) y se reintenta si
llega incompleto.

Nota (oct-2026): la API exige verificación reCAPTCHA (`require_recaptcha`). El descargador lo detecta y se
detiene con un mensaje claro; no intenta evadir esa verificación. El corpus incluido en el repo proviene
de las sentencias descargadas antes de ese cambio.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import requests

from .. import config
from .balance import load_catalogo, slugify
from .procesamiento import is_complete_pdf, parse_metadata_from_filename

logger = logging.getLogger(__name__)


class VerificacionRequerida(RuntimeError):
    """La API pidió reCAPTCHA: la descarga automática no está disponible."""


class _Api:
    def __init__(self, base_url: str = config.API_BASE_URL):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "copiloto-juridico/1.0 (proyecto academico)"

    def get(self, path: str, params: dict | None = None) -> requests.Response:
        for intento in range(config.DOWNLOAD_MAX_RETRIES):
            try:
                r = self.session.get(f"{self.base}{path}", params=params, timeout=config.DOWNLOAD_TIMEOUT)
            except requests.RequestException as e:
                espera = 2 ** intento * 5
                logger.warning("Error de red (%s). Reintentando en %ss...", e, espera)
                time.sleep(espera)
                continue
            if r.status_code == 429:
                time.sleep(int(r.headers.get("retry-after", 30)))
                continue
            if r.status_code in (401, 403) and "recaptcha" in r.text.lower():
                raise VerificacionRequerida(
                    "La API del STJ Jalisco exige verificación reCAPTCHA. La descarga automática no es posible; "
                    "copia los PDFs a la carpeta de PDFs y usa 'Procesar PDFs' en la página Datos (o el comando construir-corpus --pdfs).")
            r.raise_for_status()
            return r
        raise RuntimeError(f"No se pudo obtener {path} tras {config.DOWNLOAD_MAX_RETRIES} intentos")


def nombre_archivo(toca: dict) -> str | None:
    """Reproduce la nomenclatura original: fecha_materia_delito[_accion_especifica][_accion_simultanea].pdf"""
    try:
        partes = [toca["fecha_publicacion"], toca["materia_data"]["nombre"], toca["delito_data"]["nombre"]]
    except (KeyError, TypeError):
        return None
    for extra in ("accion_especifica_data", "accion_simultanea_data"):
        if (toca.get(extra) or {}).get("nombre"):
            partes.append(toca[extra]["nombre"])
    if not all(partes):
        return None
    return slugify("_".join(partes)) + ".pdf"


def descargar_balanceado(destino: Path = config.RAW_PDF_DIR, log: Callable[[str], None] = print,
                         ya_en_corpus: Counter | None = None) -> Counter:
    """Descarga hasta que todos los tipos del catálogo alcanzan su meta. Devuelve el conteo por tipo."""
    destino.mkdir(parents=True, exist_ok=True)
    catalogo = load_catalogo()
    metas = dict(zip(catalogo["tipo"], catalogo["meta"]))
    conteo: Counter = Counter(ya_en_corpus or {})
    for pdf in destino.glob("*.pdf"):
        meta = parse_metadata_from_filename(pdf.name)
        if meta:
            conteo[meta["tipo"]] += 1

    def pendientes() -> int:
        return sum(1 for t, m in metas.items() if conteo[t] < m)

    api = _Api()
    primera = api.get("/tocas", {"page": 1}).json()["data"]["tocas"]
    last_page = primera["last_page"]
    log(f"Catálogo: {len(metas)} tipos. Pendientes: {pendientes()}. Páginas en la API: {last_page}.")

    for page in range(1, last_page + 1):
        if pendientes() == 0:
            log("✅ Todos los tipos alcanzaron su meta. Descarga balanceada completa.")
            break
        tocas = primera["data"] if page == 1 else api.get("/tocas", {"page": page}).json()["data"]["tocas"]["data"]
        for toca in tocas:
            fname = nombre_archivo(toca)
            meta = parse_metadata_from_filename(fname) if fname else None
            if not meta:
                continue
            tipo = meta["tipo"]
            meta_tipo = metas.get(tipo, config.MAX_SENTENCIAS_POR_TIPO)  # tipos nuevos también se limitan
            path = destino / fname
            if conteo[tipo] >= meta_tipo or path.exists():
                continue
            for _ in range(config.DOWNLOAD_MAX_RETRIES):
                path.write_bytes(api.get(f"/toca/{toca['id']}/file").content)
                if is_complete_pdf(path):
                    conteo[tipo] += 1
                    log(f"   ⬇️ {fname} ({conteo[tipo]}/{meta_tipo} de {tipo})")
                    break
                log(f"   ⚠️ PDF incompleto, reintentando: {fname}")
            else:
                path.unlink(missing_ok=True)
        if page % 10 == 0:
            log(f"Página {page}/{last_page}. Tipos pendientes: {pendientes()}")
    return conteo

