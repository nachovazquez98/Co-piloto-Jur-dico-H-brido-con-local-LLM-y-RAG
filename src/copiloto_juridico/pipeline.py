"""Fase 3: pipeline del Co-piloto como generador de eventos.

La interfaz (Streamlit) o la consola consumen los eventos para animar cada paso, mostrar los logs
en vivo y renderizar el informe final. Flujo:

1. Extracción de conceptos clave (LLM)
2. Clasificación híbrida (Coseno + Jaccard + KEM + Concisión)
3. Búsqueda adaptativa (filtro estricto por tipo -> ampliación semántica)
4. Por candidato: QC_1 relevancia -> extracción (con fallback genérico) -> informe -> QC_2 anti-alucinación
5. Síntesis comparativa final (en streaming)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

from . import config, prompts
from .clasificador import ClasificadorHibrido, etiqueta_tipo, senales_coincidencia
from .embeddings import embed_query
from .llm import LLMClient, LLMError
from .vectorstore import Candidato, SentenciasStore

PASOS = [
    ("entidades", "Extracción de conceptos clave"),
    ("clasificacion", "Clasificación híbrida de la consulta"),
    ("busqueda", "Búsqueda adaptativa de precedentes"),
    ("analisis", "Análisis de precedentes (verificación, extracción e informe)"),
    ("sintesis", "Síntesis comparativa final"),
]

# Qué entra, qué sale y qué concepto usa cada paso (se muestra en el panel de pasos de la app).
DESCRIPCION_PASOS = {
    "entidades": "El modelo de lenguaje identifica en la consulta de 1 a 3 conceptos jurídicos: primero el delito, "
                 "con su modalidad (deducido de los hechos si no se nombra), y después el problema procesal (p. ej. "
                 "vinculación a proceso, prisión preventiva). Si no logra extraerlos, se utiliza la consulta íntegra.",
    "clasificacion": "La consulta se compara con las familias de delito del corpus penal mediante cuatro señales: "
                     "similitud semántica (coseno de embeddings), coincidencia de lemas (Jaccard), concisión de la "
                     "categoría y el voto de las sentencias más parecidas. Resultado: las categorías más probables, "
                     "en orden.",
    "busqueda": "Se buscan sentencias de las 3 familias más probables o, en su caso, de los delitos que seleccionó "
                "el usuario. Se combinan dos búsquedas: semántica (embeddings, encuentra el mismo supuesto aunque se "
                "redacte distinto) y literal (BM25, privilegia términos exactos como el nombre del delito); ambas "
                "listas se fusionan con RRF. Si el filtro arroja pocos resultados, se completa sin filtro. Hasta 30 "
                "sentencias candidatas.",
    "analisis": "Para cada candidata: QC_1 (el modelo verifica que trate el mismo delito o el mismo problema "
                "procesal), ficha estructurada de la resolución (acto impugnado, apelante, agravios y su "
                "calificación, sentido del fallo, fundamento legal) y verificaciones automáticas contra el texto "
                "(sentido del fallo según los puntos resolutivos, artículos citados, QC_2 del delito), hasta reunir "
                "el número de precedentes solicitado.",
    "sintesis": "Con reglas, se determina qué parte favoreció cada precedente (quién apeló y sentido del fallo). El modelo redacta, exclusivamente con esas fichas, qué "
                "argumentos acogió o desestimó la Sala; cada afirmación cita su precedente y se verifica. No se "
                "pronostica el resultado del asunto.",
}


@dataclass
class Evento:
    tipo: str            # paso_inicio | paso_fin | log | precedente | token | resultado | error
    paso: str | None = None
    mensaje: str = ""
    datos: dict = field(default_factory=dict)


@dataclass
class Opciones:
    num_precedentes: int = config.RAG_NUM_RESULTS_DESIRED
    max_candidatos: int = 15
    qc_relevancia: bool = field(default_factory=lambda: config.QC1_ACTIVO)  # QC_1 (ver config.QC1_ACTIVO)
    qc_post_reporte: bool = True
    # familias de delito que elige el usuario (una o varias); vacío = las elige el clasificador
    familias: tuple[str, ...] = ()


@dataclass
class Precedente:
    numero: int
    candidato: Candidato
    fuente: str
    informe: str
    extraccion: str
    fallback: bool
    veredicto_reporte: str = ""
    ambiguo_qc1: bool = False      # QC_1 respondió algo que no se pudo interpretar como SI/NO
    ambiguo_qc2: bool = False
    no_verificado_qc1: bool = False  # QC_1 lo descartó, pero se muestra porque QC_1 descartó todo
    sentido_reglas: str | None = None  # sentido del fallo leído de los resolutivos, sin LLM
    articulos_no_encontrados: list[str] = field(default_factory=list)
    verificar: list[str] = field(default_factory=list)  # marcas "verificar" para el usuario
    ficha: object | None = None  # FichaPenal cuando se generó en una llamada (FICHA_MODO = "json")

    @property
    def datos_penales(self) -> dict:
        """Acto impugnado, agravios, calificación y sentido del fallo. Si el sentido leído de los
        resolutivos difiere del que dio el LLM, manda el de los resolutivos y se marca para verificar."""
        from .fichas import datos_penales

        if self.candidato.materia != "penal":
            return {}
        d = datos_penales(self.extraccion)
        d["sentido_llm"] = d.get("sentido_fallo")
        if self.sentido_reglas:
            d["sentido_fallo"] = self.sentido_reglas
        return d

    def tarjeta(self) -> dict:
        """Lo que ve el abogado en la tarjeta del precedente (sirve igual para la ficha JSON y la de dos llamadas)."""
        from .fichas import DATO_NO_DISPONIBLE, campo, parsear_campos, parte_favorecida, resultado_para_defensa

        c, d = self.candidato, self.datos_penales
        f = self.ficha
        if f is not None:
            agravios = [(a["agravio"], a["calificacion"]) for a in f.agravios]
            fundamento = f.fundamento_legal
            identificador, delito, acto = f.identificador, f.delitos, f.acto_impugnado
        else:
            campos = parsear_campos(self.extraccion)
            lineas = [x.strip(" -*0123456789.") for x in (d.get("agravios") or "").splitlines() if x.strip()]
            agravios = [(x, "") for x in lineas if x and x != DATO_NO_DISPONIBLE]
            fundamento = [x.strip() for x in (campo(campos, "fundamento") or "").split(";") if x.strip()]
            identificador = campo(campos, "identificador", "expediente") or DATO_NO_DISPONIBLE
            delito, acto = d.get("delito") or DATO_NO_DISPONIBLE, d.get("acto_impugnado") or DATO_NO_DISPONIBLE
        return {
            "documento": c.doc_id, "fecha": c.fecha, "identificador": identificador, "delito": delito,
            "acto_impugnado": acto, "sentido_fallo": d.get("sentido_fallo") or DATO_NO_DISPONIBLE,
            "sentido_verificado": bool(self.sentido_reglas),  # leído de los puntos resolutivos, no solo de la ficha
            "agravios": agravios, "fundamento": fundamento, "verificar": list(self.verificar),
            "no_verificado_qc1": self.no_verificado_qc1, "articulos_no_encontrados": self.articulos_no_encontrados,
            "para_defensa": resultado_para_defensa(f.apelante, d.get("sentido_fallo")) if f is not None else None,
            "parte_favorecida": parte_favorecida(f.apelante, d.get("sentido_fallo")) if f is not None else None,
        }

    def markdown(self) -> str:
        c = self.candidato
        avisos = [f"QC_{n} ambiguo" for n, a in ((1, self.ambiguo_qc1), (2, self.ambiguo_qc2)) if a]
        return (
            f"### Precedente {self.numero}: {etiqueta_tipo(c.tipo)}\n\n"
            f"| Documento | Fecha | Materia | Similitud | Fuente |\n|---|---|---|---|---|\n"
            f"| `{c.doc_id}` | {c.fecha} | {c.materia.upper()} | {c.similitud:.3f} | {self.fuente} |\n\n"
            + ("> **No verificado por QC_1.**\n\n" if self.no_verificado_qc1 else "")
            + "".join(f"> Verificar: {v}\n" for v in self.verificar) + ("\n" if self.verificar else "")
            + (f"_Veredictos no interpretables: {', '.join(avisos)}._\n\n" if avisos else "")
            + f"{self.informe}\n"
        )


@dataclass
class Resultado:
    consulta: str
    categoria: str
    entidades: str
    tabla_clasificacion: pd.DataFrame
    modo_busqueda: str
    precedentes: list[Precedente]
    sintesis: str
    descartados: list[dict]
    segundos: float
    modelo: str
    veredictos_ambiguos: int = 0          # QC_1/QC_2 con respuesta no interpretable (el sistema continuó)
    aviso_cobertura: str | None = None    # "no hay precedentes del mismo tipo..." cuando aplica
    llamadas_llm: int = 0
    citas: dict | None = None             # verificación de citas de la síntesis (afirmaciones, válidas, marcadas)
    familias_elegidas: tuple[str, ...] = ()  # delitos que eligió el usuario (vacío = los eligió el clasificador)
    nota_delito: str | None = None        # el clasificador habría buscado en otra familia que la elegida

    def markdown(self) -> str:
        from .evaluacion.familias import etiqueta_familia

        partes = [
            "# Informe del Co-piloto Jurídico\n",
            f"**Consulta:** {self.consulta}\n",
            f"**Categoría detectada:** {etiqueta_tipo(self.categoria)}  ",
            *([f"**Delitos elegidos:** {', '.join(map(etiqueta_familia, self.familias_elegidas))}  "]
              if self.familias_elegidas else []),
            f"**Búsqueda:** {self.modo_busqueda}  ",
            f"**Modelo:** {self.modelo} · **Tiempo:** {self.segundos:.0f} s · "
            f"**Veredictos ambiguos:** {self.veredictos_ambiguos}\n",
            *([f"> ⚠️ {self.aviso_cobertura}\n"] if self.aviso_cobertura else []),
            *([f"> {self.nota_delito}\n"] if self.nota_delito else []),
            "---\n## Síntesis de precedentes relevantes\n",
            self.sintesis or "_No se generó síntesis._",
            "\n---\n## Apéndice: análisis detallado de precedentes\n",
        ]
        partes += [p.markdown() for p in self.precedentes]
        partes.append("\n---\n_Herramienta de apoyo basada en precedentes públicos del STJ Jalisco. "
                      "No sustituye la asesoría de un profesional del derecho._")
        return "\n".join(partes)


def qc2_determinista(prec: "Precedente", familias_clasificadas: set[str]) -> tuple[bool, str]:
    """QC_2 sin LLM para el alcance penal.

    La familia del delito que reporta la ficha debe ser una de las familias clasificadas para la consulta o la de
    la etiqueta de la sentencia; si no es ninguna, la ficha trata otro asunto (o el LLM extrajo mal) y se descarta.
    Si el delito no se reconoce o no coincide con la etiqueta, se acepta con una marca "verificar"."""
    from .evaluacion.familias import FAMILIA_OTROS, cargar_familias, etiqueta_familia
    from .fichas import datos_penales, familia_del_delito

    delito = datos_penales(prec.extraccion).get("delito") or ""
    fam = familia_del_delito(delito)
    fam_doc = cargar_familias([prec.candidato.tipo])[prec.candidato.tipo]
    if fam == FAMILIA_OTROS:
        prec.verificar.append(f"no se reconoció la familia del delito de la ficha ('{delito[:60]}')")
        return True, "delito no reconocido; se acepta para revisión"
    if fam not in familias_clasificadas | {fam_doc}:
        return False, (f"el delito de la ficha ({etiqueta_familia(fam)}) no es de las familias de la consulta ni de "
                       f"la etiqueta de la sentencia ({etiqueta_familia(fam_doc)})")
    if fam != fam_doc:
        prec.verificar.append(f"la ficha reporta {etiqueta_familia(fam)}; la sentencia está etiquetada como "
                              f"{etiqueta_familia(fam_doc)}")
    return True, f"delito de la ficha coherente ({etiqueta_familia(fam)})"


TITULO_BALANCE = "### Resultado de cada precedente"


def balance_precedentes(precedentes: list["Precedente"]) -> str:
    """Resultado de cada precedente armado con código a partir de su ficha (quién apeló, qué se apeló, sentido del
    fallo y parte favorecida): el LLM de 8B confundía al apelante al redactarlo. Vacío si ninguna ficha es
    estructurada."""
    filas = []
    for p in precedentes:
        if p.ficha is None:
            continue
        t = p.tarjeta()
        resultado = f"favorece a **{t['parte_favorecida']}**" if t["parte_favorecida"] else "parte favorecida no determinada"
        origen = "" if t["sentido_verificado"] else " (según la ficha; sin verificar en los puntos resolutivos)"
        filas.append(f"- **Precedente {p.numero}:** apeló {p.ficha.apelante.lower()} contra "
                     f"{t['acto_impugnado'][:1].lower() + t['acto_impugnado'][1:]}; la Sala {t['sentido_fallo']}{origen} → "
                     f"{resultado}.")
    return TITULO_BALANCE + "\n" + "\n".join(filas) if filas else ""


_RE_CITA = re.compile(r"\[\s*P\s?\d+(?:\s*[,;y]\s*P?\s?\d+)*\s*\]|\bprecedente\s+(\d+)\b", re.IGNORECASE)
_RE_FRASE = re.compile(r"[^.!?\n]+(?:[.!?](?!\d)|$)(?:\s*\[[^\]]*\])*", re.MULTILINE)
_RE_LINEA = re.compile(r".+")
_VERBOS_SINTESIS = {"CONFIRMA": r"confirm", "REVOCA": r"revoc", "MODIFICA": r"modific", "REPONE": r"repus|repon|reposici"}


def verificar_citas(sintesis: str, precedentes: list["Precedente"]) -> tuple[str, dict]:
    """Verificación determinista de la síntesis. Cada afirmación (viñeta completa, o frase de 30+ caracteres
    en un párrafo; los títulos no cuentan) debe citar un precedente existente ([P1], "Precedente 1"); si menciona un sentido del fallo
    (confirmó, revocó...) o una familia de delito, debe coincidir con la ficha de algún precedente citado. Las
    que fallan se marcan "verificar" (mismo estilo que las fichas). Devuelve (texto marcado, conteos)."""
    from .evaluacion.familias import FAMILIA_OTROS
    from .fichas import familia_del_delito

    por_numero = {p.numero: p for p in precedentes}
    conteo = {"afirmaciones": 0, "con_cita_valida": 0, "sin_cita": 0, "marcadas": 0}

    def revisar(m: re.Match) -> str:
        frase = m.group(0)
        texto = _RE_CITA.sub("", frase).strip(" -*•")
        if len(texto) < 30:
            return frase
        conteo["afirmaciones"] += 1
        numeros = {int(n) for c in _RE_CITA.finditer(frase) for n in re.findall(r"\d+", c.group(0))}
        if not numeros:
            conteo["sin_cita"] += 1
            return frase
        problemas = [f"no hay precedente {n}" for n in sorted(numeros - set(por_numero))]
        citados = [por_numero[n] for n in sorted(numeros & set(por_numero))]
        bajo = texto.lower()
        sentidos = {s for s, patron in _VERBOS_SINTESIS.items() if re.search(patron, bajo)}
        reales = {p.tarjeta()["sentido_fallo"] for p in citados}
        if citados and sentidos and not sentidos & reales:
            problemas.append(f"la ficha citada dice {', '.join(sorted(reales))}")
        familia = familia_del_delito(texto)
        if citados and familia != FAMILIA_OTROS:
            fams = {familia_del_delito(p.ficha.delitos if p.ficha else p.tarjeta()["delito"]) for p in citados}
            if familia not in fams:
                problemas.append("el delito no coincide con la ficha citada")
        if problemas:
            conteo["marcadas"] += 1
            return frase.rstrip() + f" _(verificar: {'; '.join(problemas)})_"
        conteo["con_cita_valida"] += 1
        return frase

    lineas = []
    for linea in sintesis.splitlines():
        if linea.lstrip().startswith("#"):
            lineas.append(linea)
        elif re.match(r"\s*(?:[-*•]|\d+\.)\s", linea):  # una viñeta es una afirmación (la cita suele ir al final)
            lineas.append(_RE_LINEA.sub(revisar, linea))
        else:
            lineas.append(_RE_FRASE.sub(revisar, linea))
    return "\n".join(lineas), conteo


def verificar_ficha(prec: "Precedente") -> None:
    """Chequeos deterministas de la ficha contra el texto de la sentencia (`fichas.verificaciones`): sentido del
    fallo de los resolutivos contra el del LLM y artículos citados que no aparecen en el texto."""
    from .fichas import verificaciones

    if prec.candidato.materia != "penal":
        return
    v = verificaciones(prec.extraccion, prec.candidato.texto)
    prec.sentido_reglas, prec.articulos_no_encontrados = v["sentido_reglas"], v["articulos_no_encontrados"]
    prec.verificar += v["verificar"]


_RE_VEREDICTO = re.compile(r"veredicto[^a-záéíóú]*\b(s[ií]|no)\b", re.IGNORECASE)


def parse_veredicto(texto: str) -> bool | None:
    """True si el verificador respondió SI, False si NO, None si no se pudo interpretar."""
    matches = _RE_VEREDICTO.findall(texto or "")
    if matches:
        return matches[-1].lower() != "no"
    ultima = (texto or "").strip().splitlines()[-1:] or [""]
    palabra = re.findall(r"\b(s[ií]|no)\b", ultima[0], re.IGNORECASE)
    return palabra[-1].lower() != "no" if palabra else None


def es_fallo(texto: str | None) -> bool:
    if not texto or len(texto.strip()) < 40:
        return True
    bajo = texto.lower()
    return any(k in bajo for k in config.FAILURE_KEYWORDS)


_RE_AGRAVIOS = re.compile(r"agravi|infundad|inoperante|fundad")
# Parte considerativa: donde la Sala, con su propia voz, cita las normas en que funda la decisión. No usa las
# fórmulas de fundamentación de la referencia ("con fundamento en"...), para no medir la referencia contra sí misma.
# Solo para el experimento: no mejoró el fundamento legal (`20261123-212426_5fc9b67_fundamento-contexto`).
_RE_CONSIDERATIVA = re.compile(r"art[ií]culos?\b|\bart\.|esta\s+(?:sala|alzada)|este\s+(?:tribunal|[oó]rgano)",
                               re.IGNORECASE)
_OMITIDO = "\n\n[... texto omitido por longitud ...]\n\n"


def recortar_contexto(texto: str, max_chars: int, patron: re.Pattern = _RE_AGRAVIOS) -> str:
    """Texto de la sentencia que cabe en la ventana del LLM. No se resume: se eligen partes.

    Si no cabe: el inicio (25 %: tribunal, toca, antecedentes), el final (25 %: resolutivos) y, en medio, la
    ventana del texto intermedio con más menciones a los agravios y su calificación ("agravio", "fundado",
    "infundado", "inoperante"), que es lo que más le sirve a la defensa y suele estar en los considerandos. Si el
    texto intermedio no habla de agravios, se usa 55 % inicio / 45 % final. `patron` cambia lo que se
    cuenta en la ventana intermedia (`_RE_CONSIDERATIVA`: experimento `fundamento-contexto`)."""
    if len(texto) <= max_chars:
        return texto
    cabeza, cola = int(max_chars * 0.25), int(max_chars * 0.25)
    medio = max_chars - cabeza - cola - 2 * len(_OMITIDO)
    ini, fin = cabeza, len(texto) - cola
    posiciones = [m.start() for m in patron.finditer(texto, ini, fin)]
    if not posiciones or medio <= 0:
        cabeza = int(max_chars * 0.55)
        return texto[:cabeza] + _OMITIDO + texto[-(max_chars - cabeza):]
    # Ventana de longitud `medio` que contiene más menciones (dos punteros sobre las posiciones ordenadas).
    mejor, inicio_mejor, j = 0, posiciones[0], 0
    for i, p in enumerate(posiciones):
        while posiciones[j] < p - medio:
            j += 1
        if i - j + 1 > mejor:
            mejor, inicio_mejor = i - j + 1, max(ini, p - medio + 1)
    inicio_mejor = max(ini, min(inicio_mejor, fin - medio))
    return (texto[:cabeza] + _OMITIDO + texto[inicio_mejor:inicio_mejor + medio] + _OMITIDO + texto[-cola:])


def limpiar_conceptos(respuesta: str, consulta: str) -> tuple[str, bool]:
    """Post-proceso de la respuesta del LLM en el paso de conceptos clave. Si el LLM no extrajo conceptos
    (respuesta vacía, "FALLA" o un texto demasiado largo para ser una lista), se usa la consulta completa.
    Devuelve (conceptos, ok)."""
    conceptos = (respuesta or "").strip()
    if not conceptos or "falla" in conceptos.lower() or len(conceptos) > 300:
        return consulta, False
    return conceptos, True


def combinar_candidatos(estrictos: list[Candidato], amplios: list[Candidato],
                        num_precedentes: int) -> tuple[list[tuple[Candidato, str]], str]:
    """Búsqueda adaptativa: primero los candidatos del filtro estricto; si son menos de 2×N, se completan con
    la búsqueda amplia (sin filtro). Devuelve la lista (candidato, fuente) y el modo de búsqueda resultante."""
    candidatos = [(c, "Filtro estricto") for c in estrictos]
    if len(estrictos) < num_precedentes * 2:
        vistos = {c.doc_id for c in estrictos}
        candidatos += [(c, "Búsqueda amplia") for c in amplios if c.doc_id not in vistos]
    modo = "Filtro estricto" if all(f == "Filtro estricto" for _, f in candidatos[:num_precedentes]) \
        else "Filtro estricto + búsqueda amplia" if estrictos else "Búsqueda amplia (sin filtro)"
    return candidatos, modo


def ordenar_por_facetas(candidatos: list[tuple[Candidato, str]], familias_doc: dict[str, str],
                        familias_consulta: set[str], acto: str | None,
                        actos_doc: dict[str, str]) -> list[tuple[Candidato, str]]:
    """Orden por facetas (config.ORDEN_CANDIDATOS = "familia_acto", solo si la consulta nombra un acto impugnado):
    primero misma familia de delito (la de la primera categoría) Y mismo acto impugnado que la consulta; después misma
    familia; después mismo acto de otra familia; al final el resto. Dentro de cada nivel, el orden de la búsqueda."""
    def nivel(c: Candidato) -> int:
        familia = familias_doc.get(c.tipo) in familias_consulta
        mismo_acto = acto is not None and actos_doc.get(c.doc_id) == acto
        return 0 if familia and mismo_acto else 1 if familia else 2 if mismo_acto else 3

    return sorted(candidatos, key=lambda cf: nivel(cf[0]))


@dataclass
class Busqueda:
    clasificacion: object                   # resultado de ClasificadorHibrido.clasificar (tabla, tipo, tokens_kem)
    categorias: list[str]                   # las que entran al filtro estricto
    filtro: list[str]                       # tipos que cubren esas categorías
    estrictos: list[Candidato]
    amplios: list[Candidato]
    candidatos: list[tuple[Candidato, str]]
    modo: str
    acto_consulta: str | None = None
    sugeridas: list[str] = field(default_factory=list)  # las del clasificador, aunque el usuario haya elegido otras


def familias_de_categoria(clasificador, categoria: str) -> set[str]:
    from .evaluacion.familias import cargar_familias

    tipos = clasificador.tipos_de(categoria)
    return set(cargar_familias(tipos).values())


def familias_disponibles(corpus: pd.DataFrame) -> dict[str, int]:
    """Familias de delito del corpus penal con su número de sentencias (para el selector de la app)."""
    from .evaluacion.familias import cargar_familias

    penal = corpus.loc[corpus["materia"] == "penal", "tipo"]
    return penal.map(cargar_familias(penal.unique().tolist())).value_counts().sort_index().to_dict()


def sugerir_familias(store, clasificador, consulta: str) -> list[str]:
    """Sugerencia inicial para el selector de delitos: las familias de las 3 primeras categorías del
    clasificador, sin llamar al LLM (la consulta hace de conceptos clave; tarda menos de un segundo). La búsqueda
    real usa los conceptos del LLM, así que puede diferir un poco."""
    q = embed_query(consulta)
    amplios = store.buscar(consulta, query_embedding=q, consulta_lexica=consulta)
    tabla = clasificador.clasificar(consulta, consulta, q, recuperados=amplios).tabla
    salida: list[str] = []
    for c in tabla["tipo"].head(config.NUM_CATEGORIAS_FILTRO):
        salida += [f for f in sorted(familias_de_categoria(clasificador, c)) if f not in salida]
    return salida


def nota_delito(clasificador, familias, sugeridas: list[str]) -> str | None:
    """Si el usuario eligió delitos y la primera categoría del clasificador es de otra familia, se le dice (puede
    haber elegido mal, o la descripción no deja ver el delito): no cambia la búsqueda."""
    from .evaluacion.familias import etiqueta_familia

    if not familias or not sugeridas:
        return None
    top = familias_de_categoria(clasificador, sugeridas[0])
    if top & set(familias):
        return None
    return (f"Delito seleccionado: {', '.join(map(etiqueta_familia, familias))}. Conforme a la descripción de los "
            f"hechos, el sistema habría buscado primero en {', '.join(sorted(map(etiqueta_familia, top)))}. Si la "
            "selección no corresponde al caso, agregue ese delito o retire el seleccionado.")


def buscar_candidatos(store, clasificador, consulta: str, entidades: str, q_emb, num_precedentes: int,
                      orden: str | None = None, familias: tuple[str, ...] | list[str] = ()) -> Busqueda:
    """Pasos 2 y 3 de la app (clasificación y búsqueda adaptativa), compartidos con la evaluación.

    `familias`: las familias de delito que eligió el usuario. Sustituyen a las 3 primeras categorías del
    clasificador en el filtro estricto, en el orden por facetas y en QC_2; la búsqueda amplia sigue completando si el
    filtro trae pocos candidatos (marcados como "Búsqueda amplia"), para no dejar sin resultados una familia chica."""
    from .metadatos import acto_de_consulta

    lexica = f"{consulta} {entidades}"
    amplios = store.buscar(consulta, query_embedding=q_emb, consulta_lexica=lexica)
    clasif = clasificador.clasificar(consulta, entidades, q_emb, recuperados=amplios)
    sugeridas = clasif.tabla["tipo"].head(config.NUM_CATEGORIAS_FILTRO).tolist()
    elegidas = set(familias)
    categorias = ([c for c in clasificador.tipos if familias_de_categoria(clasificador, c) & elegidas]
                  if elegidas else []) or sugeridas
    filtro = [t for c in categorias for t in clasificador.tipos_de(c)]
    estrictos = store.buscar(consulta, tipos=filtro, query_embedding=q_emb, consulta_lexica=lexica)
    candidatos, modo = combinar_candidatos(estrictos, amplios, num_precedentes)
    if config.RERANKER:  # experimento: reordena con el cross-encoder antes de las facetas
        from .reranker import reordenar

        candidatos = reordenar(consulta, candidatos, lambda d: store.fragmento_mas_similar(d, q_emb, 1500))
    acto = acto_de_consulta(consulta)
    # Sin acto en la consulta, el orden por facetas solo confiaría en la primera familia clasificada, y eso hizo perder
    # el caso de narcomenudeo (familia correcta en 2.º lugar): entonces se conserva el orden RRF.
    if (orden or config.ORDEN_CANDIDATOS) == "familia_acto" and categorias and acto is not None:
        from .evaluacion.familias import cargar_familias
        from .metadatos import cargar as cargar_metadatos

        fam_doc = cargar_familias(sorted({c.tipo for c, _ in candidatos} | set(filtro)))
        # con familias elegidas por el usuario, todas valen igual; si no, la primera del clasificador
        top = elegidas or {fam_doc.get(t, t) for t in clasificador.tipos_de(categorias[0])}
        actos = {d: m["acto_impugnado"] for d, m in cargar_metadatos().items()}
        candidatos = ordenar_por_facetas(candidatos, fam_doc, top, acto, actos)
    return Busqueda(clasif, categorias, filtro, estrictos, amplios, candidatos, modo, acto, sugeridas)


AVISO_OTRA_MATERIA = ("La consulta no menciona ningún delito ni etapa del procedimiento penal. El corpus comprende "
                      "únicamente resoluciones de las Salas Penales, por lo que estos precedentes probablemente no son "
                      "aplicables al asunto planteado.")
# Raíces de vocabulario penal (texto sin acentos, minúsculas): delitos, partes y actos del proceso.
_RE_PENAL = re.compile("|".join([
    r"\bdelit", r"\bimputad", r"\bacus", r"\bprocesad", r"\bsentenciad", r"\binculpad", r"\bvincul",
    r"ministerio publico", r"\bfiscal", r"\bpenal", r"\bprision", r"\bcarcel", r"\bdenunci", r"\bquerell",
    r"\bvictima", r"\bofendid", r"\bconden", r"\babsol", r"\baprehen", r"\bdetenid", r"\bdetencion",
    r"\bjuicio oral", r"\bmedida cautelar", r"\bsobresei", r"\bhomicid", r"\bfeminicid", r"\bmat(o|ar|aron|ado|ada)\b",
    r"\brob(o|a|ar|aron|ado|ada|os|ando|arle|arles|aba)\b", r"\bhurt", r"\basalt", r"\blesion", r"\bgolpe",
    r"\bviolen", r"\babus", r"\bviola(cion|r|da|do|ron)\b", r"\bsexual", r"\bfraud", r"\bestaf", r"\bdespoj",
    r"\bsecuestr", r"\bprivacion de la libertad", r"\bnarco", r"\bdroga", r"\bestupefac", r"\bmarihuana", r"\barma",
    r"\bdispar", r"\bamenaz", r"\bextorsi", r"\babandon", r"\bsustrac", r"\btrata\b", r"\bfalsifi", r"\bfalsedad",
    r"\bcohecho", r"\bpeculado", r"\bdano en las cosas", r"\bcorrupcion de menores", r"\bpornograf", r"\bacoso",
    r"\bhostigamiento", r"\bdesaparic", r"\bencubrim", r"\bportacion", r"\bmenudeo", r"\bculpos", r"\bimprudencia",
    r"\batropell", r"\bmaltrat", r"\bcuchill", r"\bpistola", r"\bagred", r"\bagresi", r"\bmuerte", r"\bdano fisico",
    r"\breparacion del dano", r"\btocamient"]))


def aviso_materia(consulta: str) -> str | None:
    """Aviso si la consulta no parece penal (ningún delito ni término del proceso penal)."""
    import unicodedata

    if not config.AVISO_OTRA_MATERIA:
        return None
    t = unicodedata.normalize("NFKD", consulta.lower()).encode("ascii", "ignore").decode("ascii")
    return None if _RE_PENAL.search(t) else AVISO_OTRA_MATERIA


AVISO_SIN_MISMO_TIPO = "No hay precedentes del mismo tipo en el corpus; se muestran los más cercanos."
AVISO_COINCIDENCIA_DEBIL = ("Ninguna categoría coincide bien con la consulta (señal {senal} = {valor:.3f} < "
                            "{umbral:.3f}); es probable que no haya precedentes del mismo tipo.")


def aviso_cobertura(precedentes_fuentes: list[str], senales: dict[str, float],
                    umbral: float | None = None, senal: str | None = None) -> str | None:
    """Aviso explícito de que no hay precedentes del mismo tipo.

    1. Si ningún precedente válido salió del filtro estricto, los mostrados son de otras categorías (siempre).
    2. Si hay un umbral configurado (medido en la evaluación) y la señal de coincidencia de la mejor categoría
       queda por debajo, también se avisa aunque haya resultados con filtro.
    """
    umbral = config.UMBRAL_SIN_PRECEDENTES if umbral is None else umbral
    senal = senal or config.SENAL_SIN_PRECEDENTES
    if precedentes_fuentes and all(f != "Filtro estricto" for f in precedentes_fuentes):
        return AVISO_SIN_MISMO_TIPO
    if umbral is not None and senal in senales and senales[senal] < umbral:
        return AVISO_COINCIDENCIA_DEBIL.format(senal=senal, valor=senales[senal], umbral=umbral)
    return None


class Copiloto:
    def __init__(self, store: SentenciasStore, clasificador: ClasificadorHibrido, llm: LLMClient):
        self.store = store
        self.clasificador = clasificador
        self.llm = llm
        self.llamadas = 0

    def _llm(self, prompt: str, json_mode: bool = False) -> str:
        self.llamadas += 1
        return self.llm.generate(prompt, json_mode=True) if json_mode else self.llm.generate(prompt)

    def _ficha_json(self, cand: Candidato, contexto: str, logs: list[str]):
        """Ficha penal estructurada: la precalculada si existe para esta sentencia, este texto y esta versión del
        prompt y del modelo (`fichas-batch`); si no, una llamada al LLM (JSON validado, con un reintento)."""
        from .fichas import generar_ficha_json
        from .fichas_precalculadas import ficha_guardada, fila_guardada

        if config.FICHAS_PRECALCULADAS:
            guardada = ficha_guardada(cand.doc_id, cand.texto, self.llm.settings)
            if guardada is not None:
                modelo = (fila_guardada(cand.doc_id) or {}).get("modelo") or self.llm.settings.model
                logs.append(f"   -> Ficha penal precalculada con {modelo} (sin llamar al LLM).")
                return guardada
        logs.append(f"   -> Ficha penal en una llamada ({len(contexto):,} de {len(cand.texto):,} caracteres)...")
        ficha, info = generar_ficha_json(self._llm, contexto)
        if info["intentos"] > 1:
            logs.append(f"   ⚠️ Ficha no válida ({info['error'] if ficha is None else 'primer intento'}); "
                        + ("se usa la extracción en dos llamadas." if ficha is None else "se reintentó una vez."))
        return ficha

    def _analizar_candidato(self, consulta: str, cand: Candidato, fuente: str, categorias: list[str],
                            op: "Opciones") -> tuple["Precedente | None", str, list[str]]:
        """Ficha de un candidato + verificaciones + QC_2. Devuelve (precedente | None, motivo de descarte, logs)."""
        logs: list[str] = []
        max_chars = self.llm.settings.max_context_chars
        contexto = recortar_contexto(cand.texto, max_chars)
        if cand.materia == "penal" and config.FICHA_MODO == "json":
            ficha = self._ficha_json(cand, contexto, logs)
            if ficha is not None:
                prec = Precedente(0, cand, fuente, "", ficha.campos_markdown(), False, ficha=ficha)
                verificar_ficha(prec)
                prec.informe = ficha.informe_markdown(prec.articulos_no_encontrados, prec.sentido_reglas)
                return self._qc2(consulta, prec, categorias, op, logs)
        # Ficha en dos llamadas (línea base) o respaldo si la ficha JSON no fue válida.
        p_ext, p_rep = prompts.prompts_penal()
        logs.append(f"   -> Extrayendo hechos ({len(contexto):,} de {len(cand.texto):,} caracteres)...")
        hechos = self._llm(p_ext.format(contexto=contexto))
        fallback = False
        if es_fallo(hechos):
            logs.append("   ❌ Falló la extracción. Descartado.")
            return None, "Extracción", logs
        logs.append("   -> Generando informe...")
        informe = self._llm(p_rep.format(hechos=hechos))
        if es_fallo(informe):
            logs.append("   ❌ Falló la generación del informe. Descartado.")
            return None, "Informe", logs
        prec = Precedente(0, cand, fuente, informe, hechos, fallback)
        verificar_ficha(prec)
        return self._qc2(consulta, prec, categorias, op, logs)

    def _qc2(self, consulta: str, prec: "Precedente", categorias: list[str], op: "Opciones",
             logs: list[str]) -> tuple["Precedente | None", str, list[str]]:
        """QC_2. Determinista (por defecto): descarta si el delito de la ficha no es de ninguna familia clasificada
        ni de la etiqueta de la sentencia (la ficha habla de otra cosa o el LLM extrajo mal). Con "llm", el
        verificador LLM (línea base)."""
        if op.qc_post_reporte and config.QC2_MODO == "llm":
            prec.veredicto_reporte = self._llm(prompts.PROMPT_VERIFICACION_POST_REPORTE.format(
                texto_consulta=consulta, informe_generado=prec.informe))
            ok2 = parse_veredicto(prec.veredicto_reporte)
            if ok2 is False:
                logs.append("   ❌ QC_2: el informe no es coherente con la consulta (posible alucinación). Descartado.")
                return None, "QC_2 anti-alucinación", logs
            prec.ambiguo_qc2 = ok2 is None
            logs.append("   ✅ QC_2: informe coherente con la consulta." if ok2 else
                        "   ⚠️ QC_2: veredicto ambiguo, se continúa.")
        elif op.qc_post_reporte and prec.candidato.materia == "penal":
            ok2, motivo = qc2_determinista(prec, self._familias_clasificadas(categorias))
            if not ok2:
                logs.append(f"   ❌ QC_2: {motivo}. Descartado.")
                return None, "QC_2 delito de otra familia", logs
            prec.veredicto_reporte = motivo
            logs.append(f"   ✅ QC_2: {motivo}.")
        for marca in prec.verificar:
            logs.append(f"   ⚠️ Verificar: {marca}")
        return prec, "", logs

    def _familias_clasificadas(self, categorias: list[str]) -> set[str]:
        from .evaluacion.familias import cargar_familias

        tipos = [t for c in categorias for t in self.clasificador.tipos_de(c)]
        return set(cargar_familias(tipos).values())

    def _contexto_qc1(self, cand: Candidato, q_emb) -> str:
        """Texto de la sentencia que lee QC_1, según `config.QC1_CONTEXTO`: "ficha" (campos de la ficha precalculada
        vigente; sin ella, el fragmento similar), "fragmento_similar" o "inicio"."""
        if config.QC1_CONTEXTO == "ficha" and config.FICHAS_PRECALCULADAS:
            from .fichas_precalculadas import ficha_guardada

            guardada = ficha_guardada(cand.doc_id, cand.texto, self.llm.settings)
            if guardada is not None:
                return guardada.campos_markdown()[:config.QC1_MAX_CHARS]
        if config.QC1_CONTEXTO in ("fragmento_similar", "ficha"):
            return self.store.fragmento_mas_similar(cand.doc_id, q_emb, config.QC1_MAX_CHARS)
        return cand.texto[:config.QC1_MAX_CHARS]

    def analizar(self, consulta: str, opciones: Opciones | None = None) -> Iterator[Evento]:
        op = opciones or Opciones()
        t0 = time.time()
        log = lambda m, paso=None: Evento("log", paso, m)  # noqa: E731

        # 1. Entidades
        yield Evento("paso_inicio", "entidades")
        yield log("--- PASO 1: EXTRACCIÓN DE ENTIDADES CLAVE ---", "entidades")
        try:
            entidades, ok = limpiar_conceptos(
                self._llm(prompts.prompt_conceptos().format(texto_consulta=consulta)), consulta)
        except LLMError as e:
            yield Evento("error", "entidades", str(e))
            return
        if not ok:
            yield log("   ⚠️ El LLM no extrajo entidades. Se usa la consulta completa.", "entidades")
        else:
            yield log(f"   -> Entidades extraídas: '{entidades}'", "entidades")
        yield Evento("paso_fin", "entidades", entidades)

        # 2. Clasificación
        yield Evento("paso_inicio", "clasificacion")
        yield log("--- PASO 2: RE-RANKING HÍBRIDO Y SELECCIÓN DE CATEGORÍA ---", "clasificacion")
        q_emb = embed_query(consulta)
        b = buscar_candidatos(self.store, self.clasificador, consulta, entidades, q_emb, op.num_precedentes,
                              familias=op.familias)
        clasif, categorias, filtro, estrictos = b.clasificacion, b.categorias, b.filtro, b.estrictos
        candidatos, modo = b.candidatos, b.modo
        yield log(f"   -> KEM Tokens dinámicos: {clasif.tokens_kem}", "clasificacion")
        for i, r in clasif.tabla.iterrows():
            yield log(f"      {i + 1}. '{r['tipo']}' | Híbrido:{r['score_híbrido']:.4f} | Coseno:{r['coseno']:.3f} "
                      f"| Jaccard:{r['jaccard']:.2f} | KEM:{r['kem']:.2f} | Voto:{r['voto_recuperación']:.2f} "
                      f"| Penalidad:{r['penalidad']:.2f}", "clasificacion")
        yield log(f"   ✅ Categoría final seleccionada: '{clasif.tipo}'", "clasificacion")
        nota = nota_delito(self.clasificador, op.familias, b.sugeridas)
        if op.familias:
            yield log(f"   -> Delitos elegidos por el usuario: {', '.join(op.familias)} (sustituyen a las categorías "
                      "del clasificador en el filtro).", "clasificacion")
        if nota:
            yield log(f"   ⚠️ {nota}", "clasificacion")
        senales = senales_coincidencia(clasif.tabla)
        yield Evento("paso_fin", "clasificacion", etiqueta_tipo(clasif.tipo), {"tabla": clasif.tabla,
                                                                              "senales": senales})

        # 3. Búsqueda adaptativa
        yield Evento("paso_inicio", "busqueda")
        yield log("--- PASO 3: BÚSQUEDA ADAPTATIVA (embeddings + BM25, fusión RRF) ---", "busqueda")
        yield log(f"Filtro estricto (top {len(categorias)} categorías: {', '.join(categorias)}"
                  f"{f'; {len(filtro)} tipos' if len(filtro) != len(categorias) else ''}): "
                  f"{len(estrictos)} sentencias candidatas.", "busqueda")
        if len(candidatos) > len(estrictos):
            yield log(f"   ⚠️ Pocos candidatos con el filtro. La búsqueda amplia añade "
                      f"{len(candidatos) - len(estrictos)}.", "busqueda")
        if config.ORDEN_CANDIDATOS == "familia_acto":
            yield log(f"   -> Orden por facetas: misma familia y acto impugnado ({b.acto_consulta or 'no detectado'}).",
                      "busqueda")
        for c, fuente in candidatos[:8]:
            yield log(f"   · {c.doc_id} | RRF {c.relevancia:.4f} | coseno {c.similitud:.3f} | {fuente}", "busqueda")
        if not candidatos:
            yield Evento("error", "busqueda", "No se encontraron precedentes en la base de datos.")
            return
        yield Evento("paso_fin", "busqueda", f"{len(candidatos)} candidatos · {modo}")

        # 4. Análisis de precedentes
        yield Evento("paso_inicio", "analisis")
        yield log(f"--- PASO 4: ANÁLISIS DE PRECEDENTES (meta: {op.num_precedentes} informes válidos) ---", "analisis")
        precedentes: list[Precedente] = []
        descartados: list[dict] = []
        rechazados_qc1: list[tuple[Candidato, str, str]] = []
        ambiguos = 0
        try:
            for n, (cand, fuente) in enumerate(candidatos[:op.max_candidatos], 1):
                if len(precedentes) >= op.num_precedentes:
                    break
                yield log(f"\n--- Candidato {n}: {cand.doc_id} | {cand.tipo} | similitud {cand.similitud:.3f} "
                          f"({fuente})", "analisis")
                yield Evento("progreso", "analisis", f"Analizando candidato {n}: {etiqueta_tipo(cand.tipo)}",
                             {"validos": len(precedentes), "meta": op.num_precedentes})
                veredicto_rel, ambiguo_qc1 = "", False
                if op.qc_relevancia:
                    veredicto_rel = self._llm(prompts.prompt_qc1().format(
                        texto_consulta=consulta, fragmento_documento=self._contexto_qc1(cand, q_emb)))
                    ok = parse_veredicto(veredicto_rel)
                    if ok is False:
                        yield log("   ❌ QC_1: el documento no sirve como precedente para la consulta. Descartado.",
                                  "analisis")
                        descartados.append({"documento": cand.doc_id, "tipo": cand.tipo, "motivo": "QC_1 relevancia"})
                        rechazados_qc1.append((cand, fuente, veredicto_rel))
                        continue
                    ambiguo_qc1 = ok is None
                    ambiguos += ambiguo_qc1
                    yield log("   ✅ QC_1: relevancia confirmada." if ok else
                              "   ⚠️ QC_1: veredicto ambiguo, se continúa.", "analisis")
                prec, motivo, logs = self._analizar_candidato(consulta, cand, fuente, categorias, op)
                for m in logs:
                    yield log(m, "analisis")
                if prec is None:
                    descartados.append({"documento": cand.doc_id, "tipo": cand.tipo, "motivo": motivo})
                    continue
                prec.numero, prec.ambiguo_qc1 = len(precedentes) + 1, ambiguo_qc1
                ambiguos += prec.ambiguo_qc2
                precedentes.append(prec)
                yield log(f"   ✅ Informe válido ({len(precedentes)}/{op.num_precedentes}).", "analisis")
                yield Evento("precedente", "analisis", datos={"precedente": prec})

            if not precedentes and rechazados_qc1:
                # QC_1 descartó todo: en lugar de no mostrar nada, se muestran los mejores candidatos marcados.
                yield log(f"\n   ⚠️ QC_1 descartó todos los candidatos. Se muestran los {op.num_precedentes} mejores "
                          "marcados como NO VERIFICADOS por QC_1.", "analisis")
                for cand, fuente, veredicto_rel in rechazados_qc1:
                    if len(precedentes) >= op.num_precedentes:
                        break
                    prec, motivo, logs = self._analizar_candidato(consulta, cand, fuente, categorias, op)
                    for m in logs:
                        yield log(m, "analisis")
                    if prec is None:
                        continue
                    prec.numero, prec.no_verificado_qc1 = len(precedentes) + 1, True
                    descartados = [d for d in descartados if d["documento"] != cand.doc_id]
                    precedentes.append(prec)
                    yield Evento("precedente", "analisis", datos={"precedente": prec})
        except LLMError as e:
            yield Evento("error", "analisis", str(e))
            return

        if not precedentes:
            yield Evento("error", "analisis", "No se pudo generar ninguna ficha válida (fallaron la extracción o los "
                                              "controles). Prueba a reformular la consulta con más detalle.")
            return
        yield Evento("paso_fin", "analisis", f"{len(precedentes)} informes válidos · {len(descartados)} descartados"
                     + (f" · {ambiguos} veredictos ambiguos" if ambiguos else ""))
        aviso = aviso_materia(consulta) or aviso_cobertura([p.fuente for p in precedentes], senales)
        if aviso:
            yield log(f"   ⚠️ {aviso}", "analisis")

        # 5. Síntesis
        yield Evento("paso_inicio", "sintesis")
        sintesis = ""
        if len(precedentes) < 2:
            yield log("--- PASO 5: SÍNTESIS OMITIDA (se necesitan al menos 2 precedentes) ---", "sintesis")
            sintesis = ("_Solo se encontró un precedente válido; no es posible un análisis comparativo. "
                        "Consulta el informe detallado en el apéndice._")
        else:
            yield log(f"--- PASO 5: SINTETIZANDO {len(precedentes)} INFORMES ---", "sintesis")
            informes = "\n\n---\n\n".join(
                f"Precedente {p.numero} ({p.candidato.doc_id}):\n{p.informe}" for p in precedentes)
            try:
                for token in self.llm.stream(prompts.prompt_sintesis().format(
                        text_consulta=consulta, informes_precedentes=informes)):
                    sintesis += token
                    yield Evento("token", "sintesis", token)
            except LLMError as e:
                yield Evento("error", "sintesis", str(e))
                return
            yield log("   ✅ Síntesis generada." if not es_fallo(sintesis) else
                      "   ⚠️ La síntesis parece incompleta; consulte los informes individuales.", "sintesis")
        citas = None
        if config.VERIFICAR_CITAS and len(precedentes) >= 2:
            sintesis, citas = verificar_citas(sintesis, precedentes)
            marcadas = f"; {citas['marcadas']} marcadas para verificar" if citas["marcadas"] else ""
            yield log(f"   -> Citas: {citas['con_cita_valida']}/{citas['afirmaciones']} afirmaciones con cita válida"
                      f"{marcadas}.", "sintesis")
        balance = balance_precedentes(precedentes)
        if balance and config.PROMPT_SINTESIS_VERSION != "v10":  # salvo la línea base, la síntesis parte de la lista por reglas
            sintesis = balance + "\n\n" + sintesis.strip()
        yield Evento("paso_fin", "sintesis")

        resultado = Resultado(consulta, clasif.tipo, entidades, clasif.tabla, modo, precedentes, sintesis.strip(),
                              descartados, time.time() - t0, self.llm.settings.model, ambiguos, aviso,
                              self.llamadas + (1 if len(precedentes) >= 2 else 0), citas, tuple(op.familias), nota)
        yield log(f"\nTiempo total: {resultado.segundos:.1f} s")
        yield Evento("resultado", datos={"resultado": resultado})
