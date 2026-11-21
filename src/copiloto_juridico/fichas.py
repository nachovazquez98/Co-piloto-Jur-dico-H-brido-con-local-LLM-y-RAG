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


# v11.2 (evaluacion/errores/articulos_no_encontrados.md): en la sentencia, los artículos se citan en listas largas
# ("artículos 20 fracción i, 133 fracción iii, 461, 467 fracción vii, 475"), con otras palabras ("arábigos",
# "diversos", "en relación al 219") y con palabras partidas por el PDF ("art ículo s"). Del lado de la sentencia se
# toman los números que siguen a esas palabras (hasta un punto y coma o punto, como máximo `_VENTANA` caracteres) y
# los que van justo antes de "del código", "de la ley", "constitucional"...
_RE_CLAVE_TOLERANTE = re.compile(
    r"\ba\s?r\s?t\s?[ií]\s?c\s?u\s?l\s?o\s?s?\b|\barts?\.|\bn\s?u\s?m\s?e\s?r\s?a\s?l(?:\s?e\s?s)?\b|\bordinal(?:es)?\b"
    r"|\bar[aá]bigos?\b|\bpreceptos?\b|\bdispositivos?\b|\bdiversos?\b|\bsimilares\b"
    r"|\ben\s+(?:relaci[oó]n|contexto)\s+(?:al|con\s+el|con\s+los)\b", re.IGNORECASE)
_VENTANA = 160
_RE_ANTES_DE_LEY = re.compile(r"(\d+)\s*(bis|ter|quater)?\s*,?\s*(?:de\s?l|de\s+la)\s+(?:c\s?[oó]\s?d\s?i\s?g\s?o|ley|"
                              r"constituci[oó]n|enjuiciamiento)|(\d+)\s*(bis|ter|quater)?\s+constitucional",
                              re.IGNORECASE)
_RE_NUM_LISTA = re.compile(r"(?<![\d/.-])(\d{1,4})(?![\d/])\s*(?:[°º]\s*)?(bis|ter|quater)?(?!\s*(?:de\s+(?:enero|febrero|marzo|"
                           r"abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)|años?|d[ií]as|meses))",
                           re.IGNORECASE)


def articulos_en_sentencia(texto: str) -> set[str]:
    """Artículos que la sentencia cita, con la lectura tolerante descrita arriba (incluye `extraer_articulos`)."""
    out = extraer_articulos(texto)
    for m in _RE_CLAVE_TOLERANTE.finditer(texto or ""):
        ventana = re.split(r"[;.](?!\s*\d)", texto[m.end():m.end() + _VENTANA], maxsplit=1)[0]
        out |= {n.lstrip("0") + (s or "").lower() for n, s in _RE_NUM_LISTA.findall(ventana)}
    for m in _RE_ANTES_DE_LEY.finditer(texto or ""):
        n, s = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        out.add(n.lstrip("0") + (s or "").lower())
    return out


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
# v11.2 (muestra de desacuerdos y de sentencias sin sentido, evaluacion/errores/sentido_fallo.md): también "lo
# procedente es confirmar". El verbo sin "se" ("este órgano colegiado revoca la resolución") solo cuenta si no hay
# ninguno de los anteriores: en la sección también se narra ("el colegiado confirma el amparo").
_VERBOS_SENTIDO = [
    ("REPONE", r"seordenalareposicion|seordenareponer|serepone[n]?el|sedecretalareposicion|procedenteesreponer"),
    ("REVOCA", r"serevoca[n]?|esderevocarse|revocase|procederevocar|sedeclaralarevocacion|procedenteesrevocar"),
    ("MODIFICA", r"semodifica[n]?|esdemodificarse|modificase|proceedemodificar|procedemodificar"
                 r"|procedenteesmodificar"),
    ("CONFIRMA", r"seconfirma[n]?|esdeconfirmarse|confirmase|procedeconfirmar|procedenteesconfirmar"),
]
_RE_VERBO = re.compile("|".join(f"(?P<{s}>{p})" for s, p in _VERBOS_SENTIDO))
_RE_VERBO_SIN_SE = re.compile("|".join(f"(?P<{s}>{v}[n]?(?:la|el|lo)(?!amparo|ejecutoria))" for s, v in
                                       (("REVOCA", "revoca"), ("MODIFICA", "modifica"), ("CONFIRMA", "confirma"))))
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
    marcas = list(_RE_MARCA.finditer(tramo))
    for patron in (_RE_VERBO, _RE_VERBO_SIN_SE):
        for m in reversed(marcas):
            verbos = [v.lastgroup for v in patron.finditer(tramo, m.end(), m.end() + seccion)]
            if verbos:
                return "REPONE" if "REPONE" in verbos else verbos[0]
    return None


def verificar_articulos(texto_ficha: str, texto_sentencia: str) -> list[str]:
    """Artículos citados en la ficha que NO aparecen en el texto de la sentencia (posibles alucinaciones).

    Se comparan números normalizados ("art. 161", "artículo 161", "numeral 161", "ordinal 161" -> "161"). Ojo: en
    PDFs de más de 50 páginas el corpus solo tiene 40 páginas, así que "no encontrado" significa "no está en el
    texto disponible"; por eso se marca para verificar en lugar de borrarlo.
    """
    return sorted(extraer_articulos(texto_ficha) - articulos_en_sentencia(texto_sentencia))


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


def _objeto_json(respuesta: str) -> dict:
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
    return d


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
    def desde_json(cls, respuesta: str, complemento: dict | None = None) -> "FichaPenal":
        """Valida la respuesta del LLM. Lanza ValueError con un mensaje útil para pedir la corrección.

        Acepta texto alrededor del JSON (toma del primer "{" al último "}"). Obligatorio: un sentido del fallo
        interpretable y los campos 'delitos' y 'resumen'; el resto, si falta, queda como dato no disponible.
        `complemento`: campos pedidos aparte que llenan los que la respuesta dejó vacíos."""
        d = _objeto_json(respuesta)
        for k, v in (complemento or {}).items():
            if _texto(d.get(k)) == DATO_NO_DISPONIBLE:
                d[k] = v
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

    def a_json(self) -> str:
        """La ficha validada como JSON (así se guarda precalculada)."""
        from dataclasses import asdict

        return _json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def desde_guardada(cls, texto_json: str) -> "FichaPenal":
        return cls(**_json.loads(texto_json))

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

    def informe_markdown(self, articulos_no_encontrados: list[str] | None = None,
                         sentido_resolutivos: str | None = None) -> str:
        """Informe legible (lo que antes redactaba una segunda llamada al LLM), armado con código. Si el sentido
        leído de los resolutivos difiere del de la ficha, manda el de los resolutivos y se marca."""
        faltan = set(articulos_no_encontrados or [])
        sentido = self.sentido_fallo
        if sentido_resolutivos and sentido_resolutivos != sentido:
            sentido = f"{sentido_resolutivos} (según los puntos resolutivos; el modelo leyó {sentido}: verificar)"
        para_defensa = resultado_para_defensa(self.apelante, sentido_resolutivos or self.sentido_fallo)

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
            f"- **Sentido del fallo:** {sentido}",
            f"- **Resultado para la defensa:** {para_defensa or DATO_NO_DISPONIBLE}", "",
            "#### Agravios y su calificación", agravios, "",
            "#### Hechos y decisión", f"**Hechos:** {self.hechos_clave}", "", f"**Decisión:** {self.decision}", "",
            "#### Fundamento legal",
            "\n".join(f"- {fundamento(f)}" for f in self.fundamento_legal) or f"- {DATO_NO_DISPONIBLE}",
            "", "#### Pruebas decisivas",
            "\n".join(f"- {x}" for x in self.pruebas_decisivas) or f"- {DATO_NO_DISPONIBLE}",
        ])


_DEFENSA = ("defens", "imputad", "procesad", "sentenciad", "acusad", "inculpad")
_ACUSACION = ("ministerio", "fiscal", "victima", "ofendid", "asesor")
_RESULTADO_APELANTE = {"CONFIRMA": "desfavorable", "REVOCA": "favorable", "MODIFICA": "parcialmente favorable",
                       "REPONE": "favorable (se repone el procedimiento)"}
_RESULTADO_CONTRARIO = {"CONFIRMA": "favorable", "REVOCA": "desfavorable", "MODIFICA": "parcialmente desfavorable",
                        "REPONE": "desfavorable (se repone el procedimiento)"}


def resultado_para_defensa(apelante: str, sentido: str | None) -> str | None:
    """¿El precedente le fue bien o mal a la defensa? Se deduce con reglas, no con el LLM: si apeló la defensa,
    CONFIRMA es desfavorable y REVOCA favorable; si apeló el Ministerio Público o la víctima, al revés. None si no
    se sabe quién apeló, apelaron ambos o el sentido es OTRO."""
    a = normalizar(apelante or "")
    defensa, acusacion = any(x in a for x in _DEFENSA), any(x in a for x in _ACUSACION)
    if defensa == acusacion or sentido not in _RESULTADO_APELANTE:
        return None
    return (_RESULTADO_APELANTE if defensa else _RESULTADO_CONTRARIO)[sentido]


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


def generar_ficha_json(generate, contexto: str) -> tuple["FichaPenal | None", dict]:
    """Ficha penal en una llamada (JSON validado); si no se puede interpretar, un reintento que le dice al LLM el
    error. `generate(prompt, json_mode=True)` es la función del cliente. Devuelve (ficha | None, info) con
    info = {"intentos": 1 | 2, "error": mensaje del último error}."""
    from . import prompts

    prompt = prompts.PROMPT_FICHA_PENAL_JSON.format(contexto=contexto)
    try:
        return FichaPenal.desde_json(generate(prompt, json_mode=True)), {"intentos": 1, "error": ""}
    except ValueError as e:
        error = str(e)
    respuesta = generate(prompt + f"\n\nTu respuesta anterior no era válida ({error}). Responde solo con el JSON "
                         "completo y correcto.", json_mode=True)
    try:
        return FichaPenal.desde_json(respuesta), {"intentos": 2, "error": error}
    except ValueError as e2:
        error = str(e2)
    # Tercer intento: si solo faltan campos obligatorios, se piden aparte y se unen a la respuesta anterior
    faltan = [k for k in prompts.CAMPOS_FICHA_FALTANTES if f"'{k}'" in error] if "faltan los campos" in error else []
    if not faltan:
        return None, {"intentos": 2, "error": error}
    pedido = prompts.PROMPT_FICHA_FALTANTES.format(
        campos=",\n".join(prompts.CAMPOS_FICHA_FALTANTES[k] for k in faltan), contexto=contexto)
    try:
        return FichaPenal.desde_json(respuesta, _objeto_json(generate(pedido, json_mode=True))), \
            {"intentos": 3, "error": error}
    except ValueError as e3:
        return None, {"intentos": 3, "error": str(e3)}


def verificaciones(texto_ficha: str, texto_sentencia: str) -> dict:
    """Chequeos deterministas de una ficha contra su sentencia (sin LLM): sentido del fallo leído de los
    resolutivos, sentido que dio el LLM, artículos citados que no aparecen en el texto y las marcas "verificar"
    que ve el abogado."""
    reglas = sentido_por_reglas(texto_sentencia)
    llm = datos_penales(texto_ficha).get("sentido_fallo")
    faltan = verificar_articulos(texto_ficha, texto_sentencia)
    marcas = []
    if reglas and llm and llm != reglas:
        marcas.append(f"sentido del fallo: la ficha dice {llm}, los resolutivos dicen {reglas}")
    if faltan:
        marcas.append("artículos no encontrados en la sentencia: " + ", ".join(faltan))
    return {"sentido_reglas": reglas, "sentido_llm": llm, "articulos_no_encontrados": faltan, "verificar": marcas}
