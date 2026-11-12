"""Lectura estructurada de las fichas que genera el LLM (campos, sentido del fallo, artículos citados).

La extracción del LLM es texto con viñetas del tipo `- **Delito(s) Principal(es):** Robo calificado`. Estas
funciones convierten ese texto en datos comparables, para mostrarlos en la interfaz y para evaluar la ficha
contra una anotación humana (sección 5.4 del informe).
"""

from __future__ import annotations

import re
import unicodedata

SENTIDOS = ("CONFIRMA", "MODIFICA", "REVOCA", "REPONE", "OTRO")
CALIFICACIONES_AGRAVIOS = ("FUNDADOS", "INFUNDADOS", "INOPERANTES")
DATO_NO_DISPONIBLE = "[DATO NO DISPONIBLE]"

_RE_CAMPO = re.compile(r"^\s*[-*•]?\s*\*\*(?P<campo>[^*]+?)\s*:?\s*\*\*\s*:?\s*(?P<valor>.*)$")


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y con espacios simples (para comparar textos de forma robusta)."""
    t = unicodedata.normalize("NFKD", (texto or "").lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def parsear_campos(texto: str) -> dict[str, str]:
    """{campo normalizado: valor} de una ficha en formato `- **Campo:** valor`.

    Un valor puede seguir en las líneas siguientes (p. ej. una lista de agravios); se acumula hasta el siguiente
    campo. Las claves se normalizan: `**Delito(s) Principal(es):**` -> `delito s principal es`.
    """
    campos: dict[str, str] = {}
    actual = None
    for linea in (texto or "").splitlines():
        m = _RE_CAMPO.match(linea)
        if m:
            actual = normalizar(m.group("campo"))
            campos[actual] = m.group("valor").strip()
        elif actual and linea.strip() and not linea.lstrip().startswith("#"):
            campos[actual] = (campos[actual] + "\n" + linea.strip()).strip()
    return campos


def campo(campos: dict[str, str], *prefijos: str) -> str | None:
    """Primer valor cuyo nombre de campo empieza con alguno de los prefijos (ya normalizados)."""
    for clave, valor in campos.items():
        if any(clave.startswith(p) for p in prefijos):
            return valor
    return None


_RE_SENTIDO = re.compile(r"\b(confirm\w*|modific\w*|revoc\w*|repon\w*|reposicion)\b")


def normalizar_sentido(texto: str | None) -> str | None:
    """Sentido del fallo en valores cerrados: CONFIRMA / MODIFICA / REVOCA / REPONE / OTRO.

    Si el texto ya trae un valor cerrado (prompt v11) se usa; si es texto libre (prompt v10, "Se MODIFICA la
    sentencia...") se toma el primer verbo de decisión que aparece. None si no hay dato.
    """
    if not texto or DATO_NO_DISPONIBLE.lower() in texto.lower():
        return None
    t = normalizar(texto)
    for s in SENTIDOS:
        if t.startswith(s.lower()):
            return s
    m = _RE_SENTIDO.search(t)
    if not m:
        return "OTRO"
    palabra = m.group(1)
    if palabra.startswith("confirm"):
        return "CONFIRMA"
    if palabra.startswith("modific"):
        return "MODIFICA"
    if palabra.startswith("revoc"):
        return "REVOCA"
    return "REPONE"


_RE_ARTICULOS = re.compile(
    r"\b(?:art[ií]culos?|arts?\.?|numeral(?:es)?|preceptos?)\s*((?:\d+\s*(?:[°ºo]\s*)?(?:bis|ter|quater)?"
    r"(?:\s*(?:,|;|\by\b|\be\b|\bal\b|-)\s*)?)+)", re.IGNORECASE)
_RE_NUMERO = re.compile(r"(\d+)\s*(?:[°ºo]\s*)?(bis|ter|quater)?", re.IGNORECASE)


def extraer_articulos(texto: str) -> set[str]:
    """Conjunto de números de artículo citados: "art. 233", "artículos 233 y 234", "numeral 168 bis".

    Se normalizan a "233", "168bis". Limitación: no distingue el ordenamiento (Código Penal de Jalisco vs.
    Código Nacional de Procedimientos Penales), así que "artículo 20" de dos códigos cuenta como el mismo.
    """
    out: set[str] = set()
    for m in _RE_ARTICULOS.finditer(texto or ""):
        for num, sufijo in _RE_NUMERO.findall(m.group(1)):
            out.add(num.lstrip("0") + (sufijo or "").lower())
    return out


def f1_conjuntos(predichos: set, verdaderos: set) -> dict[str, float]:
    """Precisión, recall y F1 entre dos conjuntos (p. ej. artículos de la ficha vs. anotados)."""
    vp = len(predichos & verdaderos)
    prec = vp / len(predichos) if predichos else (1.0 if not verdaderos else 0.0)
    rec = vp / len(verdaderos) if verdaderos else (1.0 if not predichos else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def coincide_texto(predicho: str | None, anotado: str | None) -> bool:
    """Coincidencia de un campo de texto libre (etapa, delito): iguales tras normalizar, o uno contiene al
    otro ("robo calificado" vs. "robo calificado art 236"). Criterio deliberadamente simple: el CSV de
    revisión permite que una persona corrija los casos dudosos."""
    p, a = normalizar(predicho or ""), normalizar(anotado or "")
    if not p or not a:
        return False
    return p == a or a in p or p in a


def datos_penales(extraccion: str) -> dict[str, str | None]:
    """Campos de la ficha penal que la interfaz muestra destacados (None si la ficha no los trae)."""
    c = parsear_campos(extraccion)
    sentido_txt = campo(c, "sentido del fallo") or campo(c, "decision especifica")
    return {
        "acto_impugnado": campo(c, "acto impugnado"),
        "agravios": campo(c, "agravios planteados", "agravios"),
        "calificacion_agravios": campo(c, "calificacion de los agravios", "calificacion"),
        "sentido_fallo": normalizar_sentido(sentido_txt) if sentido_txt else None,
        "etapa": campo(c, "etapa procesal", "etapa"),
        "delito": campo(c, "delito"),
    }
