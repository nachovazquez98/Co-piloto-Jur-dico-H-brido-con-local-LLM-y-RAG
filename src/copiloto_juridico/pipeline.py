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
    "entidades": "Entra la consulta; sale una lista de 1 a 3 conceptos jurídicos. El LLM actúa como extractor; si "
                 "falla, se usa la consulta completa.",
    "clasificacion": "Entran la consulta y sus conceptos; salen las 10 categorías más probables. Combina coseno de "
                     "embeddings, coincidencia de lemas (Jaccard, KEM) y el voto de las sentencias recuperadas.",
    "busqueda": "Entran las 3 mejores categorías; salen hasta 30 sentencias candidatas. Fusiona búsqueda semántica "
                "(embeddings) y léxica (BM25) con RRF; si el filtro da pocas, completa sin filtro.",
    "analisis": "Por candidato: QC_1 (¿es relevante?), extracción de hechos, ficha e QC_2 (¿la ficha es coherente?), "
                "hasta tener N fichas válidas. El LLM lee la sentencia recortada a su ventana de contexto.",
    "sintesis": "Entran las fichas válidas. Con reglas: qué precedentes fueron favorables o desfavorables para la "
                "defensa (quién apeló y sentido del fallo). Con el LLM, solo con esas fichas (RAG): qué argumentos "
                "aceptó o rechazó la Sala. No pronostica el resultado del caso.",
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
    qc_relevancia: bool = True
    qc_post_reporte: bool = True


@dataclass
class Precedente:
    numero: int
    candidato: Candidato
    fuente: str
    informe: str
    extraccion: str
    fallback: bool
    veredicto_relevancia: str = ""
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
        """Acto impugnado, agravios, calificación y sentido del fallo (prompt v11). Si el sentido leído de los
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
        from .fichas import DATO_NO_DISPONIBLE, campo, parsear_campos, resultado_para_defensa

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
            "agravios": agravios, "fundamento": fundamento, "verificar": list(self.verificar),
            "no_verificado_qc1": self.no_verificado_qc1, "articulos_no_encontrados": self.articulos_no_encontrados,
            "para_defensa": resultado_para_defensa(f.apelante, d.get("sentido_fallo")) if f is not None else None,
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

    def markdown(self) -> str:
        partes = [
            "# Informe del Co-piloto Jurídico\n",
            f"**Consulta:** {self.consulta}\n",
            f"**Categoría detectada:** {etiqueta_tipo(self.categoria)}  ",
            f"**Búsqueda:** {self.modo_busqueda}  ",
            f"**Modelo:** {self.modelo} · **Tiempo:** {self.segundos:.0f} s · "
            f"**Veredictos ambiguos:** {self.veredictos_ambiguos}\n",
            *([f"> ⚠️ {self.aviso_cobertura}\n"] if self.aviso_cobertura else []),
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


def balance_para_defensa(precedentes: list["Precedente"]) -> str:
    """Lista "a favor / en contra" armada con código a partir de cada ficha (quién apeló, qué se apeló, sentido del
    fallo): el LLM de 8B confundía al apelante al redactarla. Vacía si ninguna ficha es estructurada."""
    filas = []
    for p in precedentes:
        if p.ficha is None:
            continue
        t = p.tarjeta()
        resultado = f"**{t['para_defensa']} para la defensa**" if t["para_defensa"] else "resultado para la defensa no determinado"
        filas.append(f"- **Precedente {p.numero}:** apeló {p.ficha.apelante.lower()} contra "
                     f"{t['acto_impugnado'][:1].lower() + t['acto_impugnado'][1:]}; la Sala {t['sentido_fallo']} → "
                     f"{resultado}.")
    return "### Precedentes a favor y en contra\n" + "\n".join(filas) if filas else ""


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
_OMITIDO = "\n\n[... texto omitido por longitud ...]\n\n"


def recortar_contexto(texto: str, max_chars: int) -> str:
    """Texto de la sentencia que cabe en la ventana del LLM. No se resume: se eligen partes.

    Si no cabe: el inicio (25 %: tribunal, toca, antecedentes), el final (25 %: resolutivos) y, en medio, la
    ventana del texto intermedio con más menciones a los agravios y su calificación ("agravio", "fundado",
    "infundado", "inoperante"), que es lo que más le sirve a la defensa y suele estar en los considerandos. Si el
    texto intermedio no habla de agravios, se usa 55 % inicio / 45 % final como en v10."""
    if len(texto) <= max_chars:
        return texto
    cabeza, cola = int(max_chars * 0.25), int(max_chars * 0.25)
    medio = max_chars - cabeza - cola - 2 * len(_OMITIDO)
    ini, fin = cabeza, len(texto) - cola
    posiciones = [m.start() for m in _RE_AGRAVIOS.finditer(texto, ini, fin)]
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
        from .fichas_precalculadas import ficha_guardada

        if config.FICHAS_PRECALCULADAS:
            guardada = ficha_guardada(cand.doc_id, cand.texto, self.llm.settings)
            if guardada is not None:
                logs.append("   -> Ficha penal precalculada (sin llamar al LLM).")
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
        p_ext, p_rep = prompts.prompts_para(cand.materia)
        logs.append(f"   -> Extrayendo hechos con prompt {cand.materia.upper()} "
                    f"({len(contexto):,} de {len(cand.texto):,} caracteres)...")
        hechos = self._llm(p_ext.format(contexto=contexto))
        fallback = False
        if es_fallo(hechos):
            logs.append("   ⚠️ Falló la extracción específica. Reintentando con extracción genérica...")
            p_ext, p_rep = prompts.PROMPTS_DEFAULT
            hechos = self._llm(p_ext.format(contexto=contexto))
            fallback = True
            if es_fallo(hechos):
                logs.append("   ❌ Falló también la extracción genérica. Descartado.")
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
        verificador LLM de v10."""
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
        """Texto de la sentencia que lee QC_1, según `config.QC1_CONTEXTO`."""
        if config.QC1_CONTEXTO == "fragmento_similar":
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
        lexica = f"{consulta} {entidades}"
        amplios = self.store.buscar(consulta, query_embedding=q_emb, consulta_lexica=lexica)
        clasif = self.clasificador.clasificar(consulta, entidades, q_emb, recuperados=amplios)
        yield log(f"   -> KEM Tokens dinámicos: {clasif.tokens_kem}", "clasificacion")
        for i, r in clasif.tabla.iterrows():
            yield log(f"      {i + 1}. '{r['tipo']}' | Híbrido:{r['score_híbrido']:.4f} | Coseno:{r['coseno']:.3f} "
                      f"| Jaccard:{r['jaccard']:.2f} | KEM:{r['kem']:.2f} | Voto:{r['voto_recuperación']:.2f} "
                      f"| Penalidad:{r['penalidad']:.2f}", "clasificacion")
        yield log(f"   ✅ Categoría final seleccionada: '{clasif.tipo}'", "clasificacion")
        senales = senales_coincidencia(clasif.tabla)
        yield Evento("paso_fin", "clasificacion", etiqueta_tipo(clasif.tipo), {"tabla": clasif.tabla,
                                                                              "senales": senales})

        # 3. Búsqueda adaptativa
        yield Evento("paso_inicio", "busqueda")
        yield log("--- PASO 3: BÚSQUEDA ADAPTATIVA (embeddings + BM25, fusión RRF) ---", "busqueda")
        categorias = clasif.tabla["tipo"].head(config.NUM_CATEGORIAS_FILTRO).tolist()
        filtro = [t for c in categorias for t in self.clasificador.tipos_de(c)]
        estrictos = self.store.buscar(consulta, tipos=filtro, query_embedding=q_emb, consulta_lexica=lexica)
        yield log(f"Filtro estricto (top {len(categorias)} categorías: {', '.join(categorias)}"
                  f"{f'; {len(filtro)} tipos' if len(filtro) != len(categorias) else ''}): "
                  f"{len(estrictos)} sentencias candidatas.", "busqueda")
        candidatos, modo = combinar_candidatos(estrictos, amplios, op.num_precedentes)
        if len(candidatos) > len(estrictos):
            yield log(f"   ⚠️ Pocos candidatos con el filtro. La búsqueda amplia añade "
                      f"{len(candidatos) - len(estrictos)}.", "busqueda")
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
                prec.numero, prec.veredicto_relevancia, prec.ambiguo_qc1 = len(precedentes) + 1, veredicto_rel, ambiguo_qc1
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
                    prec.numero, prec.veredicto_relevancia, prec.no_verificado_qc1 = (len(precedentes) + 1,
                                                                                      veredicto_rel, True)
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
        aviso = aviso_cobertura([p.fuente for p in precedentes], senales)
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
                      "   ⚠️ La síntesis parece incompleta; revisa los informes individuales.", "sintesis")
        balance = balance_para_defensa(precedentes)
        if balance and config.PROMPT_SINTESIS_VERSION == "v11":
            sintesis = balance + "\n\n" + sintesis.strip()
        yield Evento("paso_fin", "sintesis")

        resultado = Resultado(consulta, clasif.tipo, entidades, clasif.tabla, modo, precedentes, sintesis.strip(),
                              descartados, time.time() - t0, self.llm.settings.model, ambiguos, aviso,
                              self.llamadas + (1 if len(precedentes) >= 2 else 0))
        yield log(f"\nTiempo total: {resultado.segundos:.1f} s")
        yield Evento("resultado", datos={"resultado": resultado})
