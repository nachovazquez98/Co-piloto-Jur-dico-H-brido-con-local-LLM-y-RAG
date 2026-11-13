"""Lectura estructurada de las fichas que genera el LLM (campos, sentido del fallo, artículos citados).

La extracción del LLM es texto con viñetas del tipo `- **Delito(s) Principal(es):** Robo calificado`. Estas
funciones convierten ese texto en datos comparables, para mostrarlos en la interfaz y para evaluar la ficha
contra una anotación humana (sección 5.4 del informe).
"""

from __future__ import annotations

import re
import unicodedata

SENTIDOS = ("CONFIRMA", "MODIFICA", "REVOCA", "REPONE", "OTRO")
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
    r"\b(?:art[ií]culos?|arts?\.?|numeral(?:es)?|ordinal(?:es)?|preceptos?|dispositivos?)\s*((?:\d+\s*(?:[°ºo]\s*)?(?:bis|ter|quater)?"
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


# --------------------------------------------------------------------------------------------------------------
# Sentido del fallo por reglas (puntos resolutivos)
# --------------------------------------------------------------------------------------------------------------

# Se buscan sobre el texto "compacto" (sin espacios ni acentos) porque la extracción del PDF parte palabras
# ("segund o", "co ntrol"). El orden importa solo para nombrar el valor; se elige el primer verbo que aparece.
_VERBOS_SENTIDO = [
    ("REPONE", r"seordenalareposicion|seordenareponer|serepone[n]?el|sedecretalareposicion"),
    ("REVOCA", r"serevoca[n]?|esderevocarse|revocase|procederevocar|sedeclaralarevocacion"),
    ("MODIFICA", r"semodifica[n]?|esdemodificarse|modificase|proceedemodificar|procedemodificar"),
    ("CONFIRMA", r"seconfirma[n]?|esdeconfirmarse|confirmase|procedeconfirmar"),
]
_RE_VERBO = re.compile("|".join(f"(?P<{s}>{p})" for s, p in _VERBOS_SENTIDO))
_RE_MARCA = re.compile(r"seresuelve|resuelve|proposiciones|puntosresolutivos|resolutivos")
# Anexo de datos testados al final del documento ("no.214 eliminado... lgppicr."): no es parte del fallo.
_RE_ANEXO = re.compile(r"no\d+eliminado[a-z0-9]{0,200}?lgppicr")


def _compacto(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", t)


def sentido_por_reglas(texto_sentencia: str, fraccion_final: float = 0.5, seccion: int = 2500) -> str | None:
    """Sentido del fallo leído de los puntos resolutivos, sin LLM: CONFIRMA / MODIFICA / REVOCA / REPONE.

    1. Sobre la mitad final del texto (sin el anexo de datos testados) se ubica la sección de resolutivos: la última
       marca "resuelve" / "proposiciones" / "resolutivos" que tenga un verbo de decisión en los siguientes
       `seccion` caracteres. (Los ordinales solos no sirven de marca: "primera instancia" aparece en el cuerpo.)
    2. Si la sección ordena la reposición del procedimiento, el sentido es REPONE (aunque también revoque);
       si no, el primer verbo de la sección (el del resolutivo PRIMERO).
    None si no se encuentra (en los PDF largos los resolutivos pueden caer en páginas no extraídas): entonces no
    hay contraste con el LLM; no es un error.
    """
    c = _RE_ANEXO.sub("", _compacto(texto_sentencia))
    # Mitad final en documentos largos; el texto completo si es corto (siempre al menos 3 secciones).
    tramo = c[max(0, min(int(len(c) * (1 - fraccion_final)), len(c) - 3 * seccion)):]
    for m in reversed(list(_RE_MARCA.finditer(tramo))):
        verbos = [v.lastgroup for v in _RE_VERBO.finditer(tramo, m.end(), m.end() + seccion)]
        if verbos:
            return "REPONE" if "REPONE" in verbos else verbos[0]
    return None


def verificar_articulos(texto_ficha: str, texto_sentencia: str) -> list[str]:
    """Artículos citados en la ficha que NO aparecen en el texto de la sentencia (posibles alucinaciones).

    Se comparan números normalizados ("art. 161", "artículo 161", "numeral 161", "ordinal 161" -> "161"). Ojo: en
    PDFs de más de 50 páginas el corpus solo tiene 40 páginas, así que "no encontrado" significa "no está en el
    texto disponible"; por eso se marca para verificar en lugar de borrarlo.
    """
    return sorted(extraer_articulos(texto_ficha) - extraer_articulos(texto_sentencia))


# --------------------------------------------------------------------------------------------------------------
# Ficha penal estructurada (una sola llamada al LLM, JSON)
# --------------------------------------------------------------------------------------------------------------

import json as _json
from dataclasses import dataclass, field

CALIFICACIONES = ("FUNDADO", "INFUNDADO", "INOPERANTE", "NO DISPONIBLE")
_CAMPOS_TEXTO = ("tribunal", "fecha_resolucion", "identificador", "etapa_procesal", "acto_impugnado", "apelante",
                 "delitos", "hechos_clave", "decision", "resumen")


def _texto(v) -> str:
    if v is None:
        return DATO_NO_DISPONIBLE
    if isinstance(v, list):
        return "; ".join(_texto(x) for x in v) or DATO_NO_DISPONIBLE
    return str(v).strip() or DATO_NO_DISPONIBLE


def _lista(v) -> list[str]:
    if v is None or v == DATO_NO_DISPONIBLE:
        return []
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[;\n]", v) if x.strip() and x.strip() != DATO_NO_DISPONIBLE]
    return [_texto(x) for x in v if _texto(x) != DATO_NO_DISPONIBLE]


@dataclass
class FichaPenal:
    """Ficha de un precedente penal para la defensa. Se valida al construirse desde el JSON del LLM."""

    tribunal: str
    fecha_resolucion: str
    identificador: str
    etapa_procesal: str
    acto_impugnado: str
    apelante: str
    delitos: str
    hechos_clave: str
    sentido_fallo: str
    decision: str
    resumen: str
    agravios: list[dict] = field(default_factory=list)
    fundamento_legal: list[str] = field(default_factory=list)
    pruebas_decisivas: list[str] = field(default_factory=list)

    @classmethod
    def desde_json(cls, respuesta: str) -> "FichaPenal":
        """Valida la respuesta del LLM. Lanza ValueError con un mensaje útil para pedir la corrección.

        Acepta texto alrededor del JSON (toma del primer "{" al último "}"). Obligatorio: un sentido del fallo
        interpretable y los campos 'delitos' y 'resumen'; el resto, si falta, queda como dato no disponible."""
        texto = (respuesta or "").strip()
        ini, fin = texto.find("{"), texto.rfind("}")
        if ini < 0 or fin <= ini:
            raise ValueError("la respuesta no contiene un objeto JSON")
        try:
            d = _json.loads(texto[ini:fin + 1])
        except _json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido ({e.msg}, posición {e.pos})") from e
        if not isinstance(d, dict):
            raise ValueError("el JSON no es un objeto")
        campos = {k: _texto(d.get(k)) for k in _CAMPOS_TEXTO}
        faltan = [k for k in ("delitos", "resumen") if campos[k] == DATO_NO_DISPONIBLE]
        if faltan:
            raise ValueError(f"faltan los campos obligatorios {faltan}")
        bruto = normalizar(_texto(d.get("sentido_fallo"))).upper()
        sentido = next((x for x in SENTIDOS if bruto.startswith(x)), None)
        if sentido is None:
            raise ValueError("'sentido_fallo' debe ser CONFIRMA, MODIFICA, REVOCA, REPONE u OTRO")
        agravios = []
        for a in d.get("agravios") or []:
            if isinstance(a, dict) and _texto(a.get("agravio")) != DATO_NO_DISPONIBLE:
                cal = normalizar(_texto(a.get("calificacion"))).upper()
                cal = next((c for c in CALIFICACIONES if cal.startswith(c[:6])), "NO DISPONIBLE")
                agravios.append({"agravio": _texto(a.get("agravio")), "calificacion": cal,
                                 "razon": _texto(a.get("razon"))})
            elif isinstance(a, str) and a.strip():
                agravios.append({"agravio": a.strip(), "calificacion": "NO DISPONIBLE", "razon": DATO_NO_DISPONIBLE})
        return cls(**campos, sentido_fallo=sentido, agravios=agravios,
                   fundamento_legal=_lista(d.get("fundamento_legal")), pruebas_decisivas=_lista(d.get("pruebas_decisivas")))

    def campos_markdown(self) -> str:
        """La ficha como lista de campos `- **Campo:** valor`, el mismo formato de la extracción en dos llamadas
        (así la lectura de campos y las verificaciones funcionan igual en ambos modos)."""
        agravios = "\n".join(f"  {i}. {a['agravio']}" for i, a in enumerate(self.agravios, 1)) or DATO_NO_DISPONIBLE
        calif = "\n".join(f"  {i}. {a['calificacion']}: {a['razon']}" for i, a in enumerate(self.agravios, 1)) \
            or DATO_NO_DISPONIBLE
        lineas = [
            f"- **Tribunal:** {self.tribunal}, {self.fecha_resolucion}",
            f"- **Identificador Judicial:** {self.identificador}",
            f"- **Etapa Procesal:** {self.etapa_procesal}",
            f"- **Acto Impugnado:** {self.acto_impugnado}",
            f"- **Apelante:** {self.apelante}",
            f"- **Delito(s) Principal(es):** {self.delitos}",
            f"- **Hechos Clave:** {self.hechos_clave}",
            f"- **Agravios Planteados:**\n{agravios}",
            f"- **Calificación de los Agravios:**\n{calif}",
            f"- **Sentido del Fallo:** {self.sentido_fallo}",
            f"- **Decisión Específica:** {self.decision}",
            f"- **Fundamento Legal:** {'; '.join(self.fundamento_legal) or DATO_NO_DISPONIBLE}",
            f"- **Pruebas Decisivas:** {'; '.join(self.pruebas_decisivas) or DATO_NO_DISPONIBLE}",
        ]
        return "\n".join(lineas)

    def informe_markdown(self, articulos_no_encontrados: list[str] | None = None) -> str:
        """Informe legible (lo que antes redactaba una segunda llamada al LLM), armado con código."""
        faltan = set(articulos_no_encontrados or [])

        def fundamento(f: str) -> str:
            ausentes = sorted(extraer_articulos(f) & faltan)
            return f + (f" _(no encontrado en la sentencia: art. {', '.join(ausentes)})_" if ausentes else "")

        agravios = "\n".join(f"- {a['agravio']} — **{a['calificacion']}**"
                             + (f": {a['razon']}" if a["razon"] != DATO_NO_DISPONIBLE else "")
                             for a in self.agravios) or f"- {DATO_NO_DISPONIBLE}"
        return "\n".join([
            "#### Resumen", self.resumen, "",
            "#### Datos del proceso",
            f"- **Tribunal:** {self.tribunal} · {self.fecha_resolucion}",
            f"- **Toca / expediente:** {self.identificador}",
            f"- **Acto impugnado:** {self.acto_impugnado} · **Apelante:** {self.apelante}",
            f"- **Delito(s):** {self.delitos}",
            f"- **Sentido del fallo:** {self.sentido_fallo}", "",
            "#### Agravios y su calificación", agravios, "",
            "#### Hechos y decisión", f"**Hechos:** {self.hechos_clave}", "", f"**Decisión:** {self.decision}", "",
            "#### Fundamento legal",
            "\n".join(f"- {fundamento(f)}" for f in self.fundamento_legal) or f"- {DATO_NO_DISPONIBLE}",
            "", "#### Pruebas decisivas",
            "\n".join(f"- {x}" for x in self.pruebas_decisivas) or f"- {DATO_NO_DISPONIBLE}",
        ])


def familia_del_delito(delitos: str) -> str:
    """Familia de delito del texto libre de la ficha ("Robo calificado, previsto por el art. 233...")."""
    from .datos.balance import slugify
    from .evaluacion.familias import FAMILIA_OTROS, _REGLAS_BUSQUEDA, asignar_familia

    slug = re.sub(r"^(?:el-|los-|la-)?(?:delitos?-(?:de-)?)?", "", slugify(delitos or ""))
    familia = asignar_familia(f"penal/{slug}").familia
    if familia != FAMILIA_OTROS:
        return familia
    for f, patron in _REGLAS_BUSQUEDA:  # el delito no está al inicio: se busca en todo el texto
        if patron.search(slug):
            return f
    return FAMILIA_OTROS
