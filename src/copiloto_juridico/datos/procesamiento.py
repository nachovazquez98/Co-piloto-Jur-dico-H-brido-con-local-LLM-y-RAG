"""Fase 1: extracción y limpieza del texto de los PDFs (migrado del notebook). Las páginas sin capa de texto
(escaneadas) se leen por OCR si Tesseract está instalado (`config.PDF_OCR`).

Las sentencias largas se muestrean (inicio + medio + final) porque no caben completas en el
contexto del LLM ni del modelo de embeddings.
"""

from __future__ import annotations

import functools
import logging
import re
import shutil
import subprocess
import tempfile
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

# Palabra que abre una sección al inicio de una línea; la sentencia puede numerar los párrafos ("1.VISTOS").
RE_STRUCTURAL_KEYWORDS_F1 = re.compile(
    r"^\s*(?:\d{1,3}\s*[.\-)]+\s*)?("
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
    if config.PDF_MAX_PAGES_FULL_READ <= 0 or total_pages <= config.PDF_MAX_PAGES_FULL_READ:
        return list(range(total_pages))
    num_start = min(config.PDF_FIXED_START_PAGES, total_pages)
    num_end = min(config.PDF_FIXED_END_PAGES, max(0, total_pages - num_start))
    middle_end = total_pages - num_end
    num_middle = min(max(0, config.PDF_TARGET_TOTAL_PAGES_SAMPLE - num_start - num_end), max(0, middle_end - num_start))
    indices = set(range(num_start)) | set(range(total_pages - num_end, total_pages))
    if num_middle > 0:
        indices |= {int(round(i)) for i in np.linspace(num_start, middle_end - 1, num=num_middle)}
    return sorted(indices)


@functools.lru_cache(maxsize=1)
def ocr_disponible() -> bool:
    """OCR utilizable: `config.PDF_OCR` no es "false" y están `pdftoppm`, `tesseract` y el modelo de español."""
    if str(config.PDF_OCR).lower() in {"false", "0", "no"} or not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return False
    try:
        idiomas = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "spa" in idiomas.split()


def ocr_pagina(pdf_path: Path, numero: int) -> str:
    """Texto de la página `numero` (desde 1) por OCR: pdftoppm la convierte en imagen y Tesseract la lee en español.
    Cadena vacía si falla."""
    with tempfile.TemporaryDirectory() as tmp:
        imagen = Path(tmp) / "pagina"
        try:
            subprocess.run(["pdftoppm", "-r", str(config.PDF_OCR_DPI), "-gray", "-png", "-singlefile", "-f", str(numero),
                            "-l", str(numero), str(pdf_path), str(imagen)], check=True, capture_output=True, timeout=120)
            return subprocess.run(["tesseract", f"{imagen}.png", "stdout", "-l", "spa"], check=True,
                                  capture_output=True, text=True, timeout=300).stdout
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("OCR fallido en pág %s de %s: %s", numero, pdf_path.name, e)
            return ""


def _texto_pagina(reader: PdfReader, pdf_path: Path, index: int) -> str:
    """Capa de texto de la página; si no la tiene (página escaneada) y hay OCR, el texto por OCR."""
    texto = reader.pages[index].extract_text() or ""
    if len(texto.strip()) < config.PDF_OCR_MIN_CARACTERES and ocr_disponible():
        texto = ocr_pagina(pdf_path, index + 1) or texto
    return texto


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
                textos.append(f"\n--- PÁG {index + 1}/{total} ---\n{_texto_pagina(reader, pdf_path, index)}")
            except Exception as e:  # una página dañada no invalida todo el PDF
                logger.warning("Error leyendo pág %s de %s: %s", index + 1, pdf_path.name, e)
            last = index
        return "\n".join(textos)
    except PdfReadError:
        logger.warning("PDF corrupto: %s", pdf_path.name)
    except Exception as e:
        logger.warning("Error inesperado con %s: %s", pdf_path.name, e)
    return ""


# Resolutivos dentro de una línea ("esta Sala resuelve con las siguientes: Proposiciones"); la extracción puede
# separar las letras ("R E S U E L V E").
RE_RESOLUTIVOS = re.compile(r"r\s?e\s?s\s?u\s?e\s?l\s?v\s?e|p\s?r\s?o\s?p\s?o\s?s\s?i\s?c\s?i\s?o\s?n\s?e\s?s|"
                            r"p\s?r\s?o\s?p\s?o\s?s\s?i\s?t\s?i\s?v\s?a\s?s|resolutivos", re.IGNORECASE)


def extract_and_clean_legal_text(raw_text: str) -> str:
    """Quita ruido (encabezados, numeración) y aísla el cuerpo de la resolución: desde la primera palabra de sección
    (si está en el primer 10 % del texto; si no, desde el inicio) hasta la fórmula final ("Notifíquese", "Así lo
    resolvió"...) posterior a los resolutivos. v11.2: antes el final se buscaba después de la última palabra de
    sección al inicio de línea, y un "Notifíquese" citado de la resolución de primera instancia dejaba hasta el 1 %
    del documento."""
    pre = RE_NOISE_F1.sub(" ", raw_text)
    matches = list(RE_STRUCTURAL_KEYWORDS_F1.finditer(pre))
    inicio = matches[0].start() if matches and matches[0].start() < 0.10 * len(pre) else 0
    desde = max([matches[-1].end() if matches else 0] + [m.end() for m in RE_RESOLUTIVOS.finditer(pre)][-1:])
    end = RE_FINAL_BOILERPLATE_F1.search(pre, pos=desde)
    core = pre[inicio: end.start() if end else len(pre)]
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
    texto = limpiar_texto(extract_and_clean_legal_text(raw))
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


# --------------------------------------------------------------------------------------------------------------
# Limpieza v11 del texto ya extraído (se aplica también al corpus guardado: no hacen falta los PDFs)
# --------------------------------------------------------------------------------------------------------------

# Cada entrada del anexo de datos testados termina en "LGPPICR" (Lineamientos Generales para la Protección de la
# Información Confidencial y Reservada); la extracción del PDF puede partir las palabras ("l gppicr", "po r ser").
_RE_FIN_ANEXO = re.compile(r"l\s?g\s?p\s?p\s?i\s?c\s?r\s?\.?")
_RE_INICIO_ENTRADA = re.compile(r"no\s?\.\s?[\d\s]{1,8}e\s?l\s?i\s?m\s?i\s?n|\[eliminado\]|el\s?documento\s?fue\s?testad|\.\s")
_FIRMA_ANEXO = ("lineamiento", "renglon", "porserundato", "quincuagesimo", "testad")
_ACENTOS = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uú]"}


def _tolerante(frase: str) -> str:
    """Regex de una frase que admite un espacio entre cualquier par de letras y vocales con o sin acento: la
    extracción del PDF parte las palabras al azar ("el p rograma", "del esta do")."""
    return r"\s?".join(_ACENTOS.get(ch, re.escape(ch)) for ch in frase.replace(" ", ""))


# Etiquetas de datos testados dentro del texto: "no.207eliminadoelnombrecompleto1", "no.6 eliminadoeldomicilio2",
# "no.12eliminadalaedad3", "no.5eliminadalainformacióncorrespondienteauna personarelacionada...penal79" (femenino,
# plural y hasta dos espacios metidos por la extracción; la etiqueta termina en su número).
_RE_ETIQUETA = re.compile(r"no\s?\.\s?\d+\s?" + _tolerante("eliminad") + r"\s?[oa]s?"
                          r"(?:[a-záéíóúñ]+(?: [a-záéíóúñ]+){0,2}?\d+|[a-záéíóúñ0-9]*)", re.IGNORECASE)
# Encabezados y pies de página que se repiten en cada hoja: "tribunal de justicia del estado de jalisco 5116 --- 115",
# la nota del programa de testado, "documento para versión electrónica", "página -12-".
_RE_ENCABEZADO = re.compile("|".join([
    r"(?:\d+\s*)?---\s*(?:\d+\s+)?",
    _tolerante("el documento fue testado con el programa elida eliminador de datos judicial") + r"(?:\s?del\b)?",
    r"(?:supremo\s?)?" + _tolerante("tribunal de justicia") + r"\s?(?:del|en\s?el)\s?" + _tolerante("estado de jalisco"),
    _tolerante("justicia del estado de jalisco sistema acusatorio adversarial"),
    _tolerante("documento para versión electrónica"),
    r"\bp[aá]gina\s?-\s?\d+\s?-|\bp[aá]gina \d+\b(?= toca)|\bp[aá]g\s?\.\s?\d+\b",
]), re.IGNORECASE)
# Número de expediente que el encabezado repite en cada hoja: "toca penal 582025", "carpeta administrativa 1234".
_RE_EXPEDIENTE = re.compile(r"\b(?:toca(?: penal| de juicio oral electr[oó]nico| juicio oral| tradicional)?"
                            r"|carpeta administrativa(?: digitalizada)?|carpeta de investigaci[oó]n|causa penal"
                            r"|averiguaci[oó]n previa)\s?\.?\s?(?:no\.\s?)?[\w./-]*\d[\w./-]*", re.IGNORECASE)
# Resto del encabezado "supremo tribunal de justicia del estado de jalisco" cuando la extracción intercala el número
# de expediente ("... 3592025 de justicia en el estado de jalisco"), y el de las salas con número de página
# ("décima tribunal de justicia del estado sistema acusatorio adversarial", "tribunal de justicia del estado 81").
# No se toca "procuraduría (general) de justicia del estado de jalisco", que sí es texto.
_RE_RESTO_ENCABEZADO = re.compile(
    r"(?P<previa>\b[a-záéíóúñ]+\s)?\bde\s?justicia\s?(?:del|en\s?el)\s?estado\s?de\s?jalisco\b"
    r"|\btribunal de justicia del estado(?: sistema acusatorio adversarial| \d{1,3}\b)", re.IGNORECASE)
# Entradas del anexo de testado que quedaron cortadas por una marca de páginas omitidas (sin "lgppicr" al final).
_RE_ANEXO_CORTADO = re.compile(r"(?:exp\.\s?)?\[eliminado\]\s?(?:[a-záéíóúñ]+\s)?en \d+ renglon(?:es)? por ser un dato "
                               r"identificativo(?: de conformidad con el lineamiento(?: [a-záéíóúñ]+){0,3})?")
# Encabezado repetido seguido (3 veces o más), con o sin número de página: "décima prime ra sala en mate ria penal 80
# décima prime ra sala en mate ria penal 81 ...". Se deja una vez. No se aplica a listas de "[eliminado]".
_RE_REPETIDO_SEGUIDO = re.compile(r"(?P<frase>(?:[^\s\[\]]+ ){1,10}?[^\s\[\]\d]{3,})(?: \d{1,4})?"
                                  r"(?: (?P=frase)(?: \d{1,4})?){2,}")
# Marcas de páginas omitidas seguidas (tras quitar el ruido entre ellas) se unen en una sola, sumando las páginas.
_RE_OMITIDAS_SEGUIDAS = re.compile(r"\.\.\. \d+ págs omitidas \.\.\.(?: \.\.\. \d+ págs omitidas \.\.\.)+")
# Palabras deletreadas por la maquetación: "r e s u e l v e" -> "resuelve" (4 o más letras sueltas seguidas).
_RE_DELETREADA = re.compile(r"\b(?:[a-záéíóúñ] ){3,}[a-záéíóúñ]\b")


def _compacto(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def quitar_anexo_testado(texto: str) -> str:
    """Quita las entradas del anexo de datos testados ("no.50 eliminadoelnombrecompleto en 1 renglones por ser un
    dato identificativo de conformidad con el lineamiento ... lgppicr."). En el corpus ocupan una mediana de ~40 % del
    texto en las sentencias que lo tienen, y desplazaban a los resolutivos del final."""
    partes, previo = [], 0
    for m in _RE_FIN_ANEXO.finditer(texto):
        segmento = texto[previo:m.end()]
        ventana_ini = max(0, len(segmento) - 450)
        inicios = [x for x in _RE_INICIO_ENTRADA.finditer(segmento, ventana_ini)]
        corte = inicios[-1].start() if inicios else ventana_ini
        if segmento[corte:corte + 2] == ". ":
            corte += 2
        if any(f in _compacto(segmento[corte:]) for f in _FIRMA_ANEXO):
            partes.append(segmento[:corte])
        else:
            partes.append(segmento)
        previo = m.end()
    partes.append(texto[previo:])
    return "".join(partes)


def _quitar_expedientes_repetidos(texto: str, minimo: int = 3) -> str:
    """Un número de expediente que aparece `minimo` veces o más es encabezado de página: se deja la primera
    aparición (de ahí se lee el identificador) y se quitan las demás."""
    claves = [re.sub(r"\s", "", m.group(0).lower()) for m in _RE_EXPEDIENTE.finditer(texto)]
    repetidas = {k for k in set(claves) if claves.count(k) >= minimo}
    vistas: set[str] = set()

    def _sub(m: re.Match) -> str:
        k = re.sub(r"\s", "", m.group(0).lower())
        if k not in repetidas or k not in vistas:
            vistas.add(k)
            return m.group(0)
        return " "

    return _RE_EXPEDIENTE.sub(_sub, texto) if repetidas else texto


def _resto_encabezado(m: re.Match) -> str:
    previa = m.group("previa") or ""
    return m.group(0) if previa.lower().startswith(("procura", "general")) else previa + " "


def _sumar_omitidas(m: re.Match) -> str:
    return f"... {sum(map(int, re.findall(r'(\d+) págs', m.group(0))))} págs omitidas ..."


def _limpiar_una_vez(t: str) -> str:
    t = quitar_anexo_testado(t)
    t = _RE_ETIQUETA.sub("[eliminado]", t)
    t = re.sub(r"\s+", " ", t)
    t = _RE_DELETREADA.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = _RE_ENCABEZADO.sub(" ", t)
    t = _RE_RESTO_ENCABEZADO.sub(_resto_encabezado, t)
    t = _RE_ANEXO_CORTADO.sub(" ", t)
    t = _quitar_expedientes_repetidos(t)
    t = re.sub(r"\s+", " ", t)
    t = _RE_REPETIDO_SEGUIDO.sub(lambda m: m.group("frase"), t)
    t = _RE_OMITIDAS_SEGUIDAS.sub(_sumar_omitidas, t)
    return t.strip()


def limpiar_texto(texto: str) -> str:
    """Limpieza v11, idempotente: anexo de datos testados, etiquetas de testado -> "[eliminado]", encabezados y
    numeración de página, palabras deletreadas y espacios repetidos. Conserva las marcas "... N págs omitidas ..."
    (avisan que esa parte del PDF no se leyó). Se repite hasta que el texto no cambia, porque unir palabras o
    quitar un encabezado puede dejar completo otro."""
    t = texto
    for _ in range(4):
        nuevo = _limpiar_una_vez(t)
        if nuevo == t:
            break
        t = nuevo
    return t
