"""Fase 1: extracción y limpieza del texto de los PDFs (migrado del notebook).

Las sentencias largas se muestrean (inicio + medio + final) porque no caben completas en el
contexto del LLM ni del modelo de embeddings.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .. import config
from .balance import tipo_key

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

RE_NOISE_F1 = re.compile(
    r"\n?\s*toca\s*[\d/]+.*?\n|"
    r"\n?\s*exp(ediente)?:?\s*[\d/a-z-]+.*?\n|"
    r"\n?\s*p[aá]gina\s*\d+.*?\n|"
    r"\n?\s*--- PÁG.*?---|"
    r"\n?\s*documento\s+para\s+versi[oó]n\s+electr[oó]nica.*?\n|"
    r"\n?\s*el\s+documento\s+fue\s+testado\s+con\s+el\s+programa\s+'elida'.*?\n|"
    r"\n?\s*supremo\s+tribunal\s+de\s+justicia\s+del\s+estado.*?\n|"
    r"\n?\s*(primera|segunda|tercera|cuarta|quinta|sexta|s[eé]ptima|octava|novena|d[eé]cima)\s+sala.*?\n",
    re.IGNORECASE | re.DOTALL,
)

RE_STRUCTURAL_KEYWORDS_F1 = re.compile(
    r"^\s*("
    r"V\s*I\s*S\s*T\s*O\s*(?:S)?\b"
    r"|R\s*E\s*S\s*U\s*L\s*T\s*A\s*N\s*D\s*O\s*(?:S)?\b"
    r"|A\s*N\s*T\s*E\s*C\s*E\s*D\s*E\s*N\s*T\s*E\s*(?:S)?\b"
    r"|C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O\s*(?:S)?\b"
    r"|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\b"
    r"|P\s*R\s*O\s*P\s*O\s*S\s*I\s*C\s*I\s*O\s*N\s*(?:E\s*S)?\b"
    r"|R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*V\s*O\s*(?:S)?\b"
    r"|SE\s+RESUELVE\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

RE_FINAL_BOILERPLATE_F1 = re.compile(
    r"(\bfundamentaci[oó]n\s+legal\b|\bNOTIF[ÍI]QUESE\b|As[ií]\s+lo\s+resolvi[oó]|Arch[ií]vese)",
    re.IGNORECASE | re.DOTALL,
)


def pages_to_read(total_pages: int) -> list[int]:
    """Páginas a leer: todas si el PDF es corto; si es largo, inicio + muestra del medio + final."""
    if total_pages <= config.PDF_MAX_PAGES_FULL_READ:
        return list(range(total_pages))
    num_start = min(config.PDF_FIXED_START_PAGES, total_pages)
    num_end = min(config.PDF_FIXED_END_PAGES, max(0, total_pages - num_start))
    middle_end = total_pages - num_end
    num_middle = min(max(0, config.PDF_TARGET_TOTAL_PAGES_SAMPLE - num_start - num_end), max(0, middle_end - num_start))
    indices = set(range(num_start)) | set(range(total_pages - num_end, total_pages))
    if num_middle > 0:
        indices |= {int(round(i)) for i in np.linspace(num_start, middle_end - 1, num=num_middle)}
    return sorted(indices)


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        if total == 0:
            return ""
        textos, last = [], -1
        for index in pages_to_read(total):
            if last != -1 and index > last + 1:
                textos.append(f"\n\n [... {index - last - 1} pág(s) omitida(s) ...] \n\n")
            try:
                textos.append(f"\n--- PÁG {index + 1}/{total} ---\n{reader.pages[index].extract_text() or ''}")
            except Exception as e:  # una página dañada no invalida todo el PDF
                logger.warning("Error leyendo pág %s de %s: %s", index + 1, pdf_path.name, e)
            last = index
        return "\n".join(textos)
    except PdfReadError:
        logger.warning("PDF corrupto: %s", pdf_path.name)
    except Exception as e:
        logger.warning("Error inesperado con %s: %s", pdf_path.name, e)
    return ""


def extract_and_clean_legal_text(raw_text: str) -> str:
    """Quita ruido (encabezados, numeración) y aísla el cuerpo de la resolución."""
    pre = RE_NOISE_F1.sub(" ", raw_text)
    matches = list(RE_STRUCTURAL_KEYWORDS_F1.finditer(pre))
    if matches:
        end = RE_FINAL_BOILERPLATE_F1.search(pre, pos=matches[-1].end())
        core = pre[matches[0].start(): end.start() if end else len(pre)]
    else:
        end = RE_FINAL_BOILERPLATE_F1.search(pre)
        core = pre[: end.start()] if end else pre
    cleaned = re.sub(r"\s+", " ", core).strip()
    cleaned = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ.,;?¿!¡\s-]", "", cleaned)
    return cleaned.lower().strip()


def parse_metadata_from_filename(filename: str) -> dict | None:
    """`2025-03-14-193036_civil_accion-de-divorcio.pdf` -> fecha, materia y delito/acción."""
    parts = Path(filename).stem.lower().split("_")
    if len(parts) < 3 or not parts[1].strip() or not "_".join(parts[2:]).strip():
        return None
    materia, accion = parts[1].strip(), "_".join(parts[2:]).strip()
    return {"fecha": parts[0][:10], "materia": materia, "delito_o_accion": accion, "tipo": tipo_key(materia, accion)}


def is_complete_pdf(path: Path) -> bool:
    """Valida que el PDF descargado esté completo: cabecera %PDF, marcador %%EOF y páginas legibles."""
    try:
        with open(path, "rb") as f:
            head = f.read(5)
            f.seek(max(0, path.stat().st_size - 2048))
            tail = f.read()
        if head != b"%PDF-" or b"%%EOF" not in tail:
            return False
        return len(PdfReader(str(path)).pages) > 0
    except Exception:
        return False


def process_single_pdf(pdf_path: str) -> dict | None:
    path = Path(pdf_path)
    meta = parse_metadata_from_filename(path.name)
    if not meta:
        return None
    raw = extract_text_from_pdf(path)
    if not raw:
        return None
    texto = extract_and_clean_legal_text(raw)
    if len(texto) < 100:
        return None
    return {"id": path.name, **meta, "texto": texto, "origen": "pdf"}


def process_pdf_folder(folder: Path, log: Callable[[str], None] = print) -> list[dict]:
    pdfs = sorted(str(p) for p in Path(folder).glob("*.pdf"))
    log(f"{len(pdfs)} PDFs encontrados en '{folder}'. Procesando en paralelo...")
    resultados, fallos = [], 0
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS_FASE_1) as ex:
        futures = {ex.submit(process_single_pdf, p): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                resultados.append(r)
            else:
                fallos += 1
            if i % 50 == 0 or i == len(pdfs):
                log(f"   -> {i}/{len(pdfs)} PDFs procesados ({fallos} fallidos)")
    return resultados
