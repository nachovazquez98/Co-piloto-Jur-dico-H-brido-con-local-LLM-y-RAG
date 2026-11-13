"""Metadatos procesales por reglas, sin LLM (v11.1): acto impugnado, quién apeló y sentido del fallo.

Casi todas las sentencias de apelación del STJ Jalisco abren con el mismo párrafo: "vistos para resolver los autos
del toca penal …, formado con motivo del recurso de apelación interpuesto por EL APELANTE, en contra de EL ACTO …".
Las reglas leen ese párrafo; el sentido sale de los puntos resolutivos (`fichas.sentido_por_reglas`). Lo que las
reglas no reconocen queda "no determinado" (no se adivina). Se calculan una vez para el corpus penal
(`copiloto-datos metadatos`) y se guardan junto al corpus.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from . import config

NO_DETERMINADO = "no determinado"
ACTOS = ("no vinculación", "vinculación", "medida cautelar", "sobreseimiento", "sentencia definitiva", "otro")
APELANTES = ("defensa", "ministerio público", "asesor de la víctima", "varios")
METADATOS_PATH = config.CORPUS_DIR / "metadatos_procesales.parquet"


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t)


def _compacto(texto: str) -> str:
    """Sin espacios ni signos: la extracción del PDF parte palabras al azar ("interpuest o por l a")."""
    return re.sub(r"[^a-z]", "", _norm(texto))


# Acto impugnado (sobre el texto compacto): el primero que coincide, en este orden ("no vinculación" contiene
# "vinculación"; si se impugnan juntas la vinculación y la prisión preventiva, cuenta como medida cautelar).
_REGLAS_ACTO = [
    ("medida cautelar", r"medidascautelar|medidacautelar|prisionpreventiva"),
    ("no vinculación", r"novincula|noavincula|nohalugaravincula|negativadevincula|nosevincul"),
    ("vinculación", r"vinculacionaproceso|autodevinculacion|vinculoaproceso|sevinculo|vincularaproceso"),
    ("sobreseimiento", r"sobresei"),
    ("sentencia definitiva", r"sentenciadefinitiva|ladefinitiva|sentenciacondenatoria|sentenciaabsolutoria|"
                             r"falloconden|falloabsolut|abreviado|secondenoa|absolvioa|lasentencia(?!interlocutoria)"),
]
_RE_ACTO = [(a, re.compile(p)) for a, p in _REGLAS_ACTO]
_RE_DEFENSA = re.compile(r"defens|imputad|procesad|sentenciad|acusad|inculpad|indiciad|enjuiciad")
_RE_MP = re.compile(r"ministeriopublico|fiscal|representantesocial|representacionsocial")
_RE_ASESOR = re.compile(r"asesor|victima|ofendid|agraviad|coadyuvante")


def _t(frase: str) -> str:
    from .datos.procesamiento import _tolerante

    return _tolerante(frase)


# "interpuesto por … en contra de …" (y variantes), tolerante a palabras partidas.
_RE_INTERPUESTO = re.compile(
    "(?:" + "|".join(_t(x) for x in ("interpuesto", "interpuesta", "interpuestos", "interpuso", "promovido",
                                    "promovida", "hecho valer"))
    + r")\s?(?:[a-z]+\s?){0,3}?" + _t("por") + r"\b(?P<apelante>.{3,350}?)(?:"
    + "|".join(_t(x) for x in ("en contra", "contra", "respecto", "en torno", "en relacion", "en relación"))
    + r")(?P<acto>.{0,600})")
_RE_ADHESIVA = re.compile(_t("adhesiva") + r"(?P<quien>.{3,200}?)(?:" + _t("contra") + "|;|$)")


def _acto(texto: str) -> str:
    c = _compacto(texto)
    for acto, patron in _RE_ACTO:
        if patron.search(c):
            return acto
    return "otro"


def _apelante(texto: str) -> str:
    c = _compacto(texto)
    quienes = [q for q, r in (("defensa", _RE_DEFENSA), ("ministerio público", _RE_MP),
                              ("asesor de la víctima", _RE_ASESOR)) if r.search(c)]
    return quienes[0] if len(quienes) == 1 else "varios" if quienes else NO_DETERMINADO


def metadatos_de_sentencia(texto: str, inicio: int = 5000) -> dict[str, str]:
    """Acto impugnado, apelante y sentido de una sentencia penal (valores cerrados o "no determinado")."""
    from .fichas import sentido_por_reglas

    cabeza = _norm(texto[:inicio])
    m = _RE_INTERPUESTO.search(cabeza)
    if m:
        acto, apelante = _acto(m.group("acto")), _apelante(m.group("apelante"))
        # la apelación adhesiva de la otra parte también cuenta ("así como la apelación adhesiva de la asesora…")
        adhesiva = _RE_ADHESIVA.search(cabeza, m.start(), m.end() + 300)
        if adhesiva and _apelante(adhesiva.group("quien")) not in (apelante, NO_DETERMINADO):
            apelante = "varios"
    else:
        acto, apelante = NO_DETERMINADO, NO_DETERMINADO
    return {"acto_impugnado": acto, "apelante": apelante, "sentido": sentido_por_reglas(texto) or NO_DETERMINADO}


def acto_de_consulta(consulta: str) -> str | None:
    """Acto impugnado que describe la consulta del abogado ("me vincularon a proceso", "prisión preventiva"); None
    si no menciona ninguno de los actos reconocidos."""
    t = re.sub(r"vincularon|vinculan|vinculado|vincular|vinculo", "vinculacion a proceso", _norm(consulta))
    acto = _acto(t)
    return None if acto == "otro" else acto


def calcular(corpus: pd.DataFrame) -> pd.DataFrame:
    penal = corpus[corpus["materia"] == "penal"]
    filas = [{"doc_id": d, **metadatos_de_sentencia(t)} for d, t in zip(penal["id"], penal["texto"])]
    return pd.DataFrame(filas, columns=["doc_id", "acto_impugnado", "apelante", "sentido"])


def guardar(tabla: pd.DataFrame, path: Path = METADATOS_PATH) -> Path:
    tabla.to_parquet(path, compression="zstd", index=False)
    return path


def cargar(path: Path = METADATOS_PATH) -> dict[str, dict[str, str]]:
    """{doc_id: {acto_impugnado, apelante, sentido}}; vacío si no se han calculado."""
    if not path.exists():
        return {}
    return pd.read_parquet(path).set_index("doc_id").to_dict("index")


def cobertura(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por campo: cuántas sentencias quedan en cada valor (incluido "no determinado")."""
    filas = []
    for campo in ("acto_impugnado", "apelante", "sentido"):
        for valor, n in tabla[campo].value_counts().items():
            filas.append({"campo": campo, "valor": valor, "sentencias": int(n), "fracción": n / len(tabla)})
    return pd.DataFrame(filas)
