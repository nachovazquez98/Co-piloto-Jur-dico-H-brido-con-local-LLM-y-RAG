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
    "sintesis": "Entran las fichas válidas; sale una comparación que responde la consulta usando solo esas fichas "
                "(RAG: el LLM no usa conocimiento propio).",
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

    @property
    def datos_penales(self) -> dict:
        """Acto impugnado, agravios, calificación y sentido del fallo, si la ficha los trae (prompt v11)."""
        from .fichas import datos_penales

        return datos_penales(self.extraccion) if self.candidato.materia == "penal" else {}

    def markdown(self) -> str:
        c = self.candidato
        avisos = [f"QC_{n} ambiguo" for n, a in ((1, self.ambiguo_qc1), (2, self.ambiguo_qc2)) if a]
        return (
            f"### Precedente {self.numero}: {etiqueta_tipo(c.tipo)}\n\n"
            f"| Documento | Fecha | Materia | Similitud | Fuente |\n|---|---|---|---|---|\n"
            f"| `{c.doc_id}` | {c.fecha} | {c.materia.upper()} | {c.similitud:.3f} | {self.fuente} |\n\n"
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


def recortar_contexto(texto: str, max_chars: int) -> str:
    """Si la sentencia no cabe en el contexto del LLM, conserva el inicio (antecedentes) y el final
    (considerandos finales y resolutivos), que es donde están los datos que se extraen."""
    if len(texto) <= max_chars:
        return texto
    cabeza = int(max_chars * 0.55)
    cola = max_chars - cabeza
    return texto[:cabeza] + "\n\n[... texto intermedio omitido por longitud ...]\n\n" + texto[-cola:]


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

    def _llm(self, prompt: str) -> str:
        self.llamadas += 1
        return self.llm.generate(prompt)

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
        max_chars = self.llm.settings.max_context_chars
        precedentes: list[Precedente] = []
        descartados: list[dict] = []
        ambiguos = 0
        for n, (cand, fuente) in enumerate(candidatos[:op.max_candidatos], 1):
            if len(precedentes) >= op.num_precedentes:
                break
            yield log(f"\n--- Candidato {n}: {cand.doc_id} | {cand.tipo} | similitud {cand.similitud:.3f} ({fuente})",
                      "analisis")
            yield Evento("progreso", "analisis", f"Analizando candidato {n}: {etiqueta_tipo(cand.tipo)}",
                         {"validos": len(precedentes), "meta": op.num_precedentes})
            try:
                veredicto_rel = ""
                ambiguo_qc1 = ambiguo_qc2 = False
                if op.qc_relevancia:
                    veredicto_rel = self._llm(prompts.PROMPT_VERIFICACION_RELEVANCIA.format(
                        texto_consulta=consulta, fragmento_documento=self._contexto_qc1(cand, q_emb)))
                    ok = parse_veredicto(veredicto_rel)
                    if ok is None:
                        ambiguos += 1
                        ambiguo_qc1 = True
                    if ok is False:
                        yield log("   ❌ QC_1: el contexto del documento NO corresponde a la consulta. Descartado.",
                                  "analisis")
                        descartados.append({"documento": cand.doc_id, "tipo": cand.tipo, "motivo": "QC_1 relevancia"})
                        continue
                    yield log("   ✅ QC_1: relevancia confirmada." if ok else
                              "   ⚠️ QC_1: veredicto ambiguo, se continúa.", "analisis")

                p_ext, p_rep = prompts.prompts_para(cand.materia)
                contexto = recortar_contexto(cand.texto, max_chars)
                yield log(f"   -> Extrayendo hechos con prompt {cand.materia.upper()} "
                          f"({len(contexto):,} de {len(cand.texto):,} caracteres)...", "analisis")
                hechos = self._llm(p_ext.format(contexto=contexto))
                fallback = False
                if es_fallo(hechos):
                    yield log("   ⚠️ Falló la extracción específica. Reintentando con extracción genérica...",
                              "analisis")
                    p_ext, p_rep = prompts.PROMPTS_DEFAULT
                    hechos = self._llm(p_ext.format(contexto=contexto))
                    fallback = True
                    if es_fallo(hechos):
                        yield log("   ❌ Falló también la extracción genérica. Descartado.", "analisis")
                        descartados.append({"documento": cand.doc_id, "tipo": cand.tipo, "motivo": "Extracción"})
                        continue

                yield log("   -> Generando informe...", "analisis")
                informe = self._llm(p_rep.format(hechos=hechos))
                if es_fallo(informe):
                    yield log("   ❌ Falló la generación del informe. Descartado.", "analisis")
                    descartados.append({"documento": cand.doc_id, "tipo": cand.tipo, "motivo": "Informe"})
                    continue

                veredicto_rep = ""
                if op.qc_post_reporte:
                    veredicto_rep = self._llm(prompts.PROMPT_VERIFICACION_POST_REPORTE.format(
                        texto_consulta=consulta, informe_generado=informe))
                    ok2 = parse_veredicto(veredicto_rep)
                    if ok2 is False:
                        yield log("   ❌ QC_2: el informe no es coherente con la consulta (posible alucinación). "
                                  "Descartado.", "analisis")
                        descartados.append({"documento": cand.doc_id, "tipo": cand.tipo, "motivo": "QC_2 anti-alucinación"})
                        continue
                    if ok2 is None:
                        ambiguos += 1
                        ambiguo_qc2 = True
                    yield log("   ✅ QC_2: informe coherente con la consulta." if ok2 else
                              "   ⚠️ QC_2: veredicto ambiguo, se continúa.", "analisis")
            except LLMError as e:
                yield Evento("error", "analisis", str(e))
                return

            prec = Precedente(len(precedentes) + 1, cand, fuente, informe, hechos, fallback, veredicto_rel, veredicto_rep,
                              ambiguo_qc1, ambiguo_qc2)
            precedentes.append(prec)
            yield log(f"   ✅ Informe válido ({len(precedentes)}/{op.num_precedentes}).", "analisis")
            yield Evento("precedente", "analisis", datos={"precedente": prec})

        if not precedentes:
            yield Evento("error", "analisis", "Ningún precedente superó los controles de calidad. "
                                              "Prueba a reformular la consulta con más detalle.")
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
                for token in self.llm.stream(prompts.PROMPT_SINTESIS_FINAL.format(
                        text_consulta=consulta, informes_precedentes=informes)):
                    sintesis += token
                    yield Evento("token", "sintesis", token)
            except LLMError as e:
                yield Evento("error", "sintesis", str(e))
                return
            yield log("   ✅ Síntesis generada." if not es_fallo(sintesis) else
                      "   ⚠️ La síntesis parece incompleta; revisa los informes individuales.", "sintesis")
        yield Evento("paso_fin", "sintesis")

        resultado = Resultado(consulta, clasif.tipo, entidades, clasif.tabla, modo, precedentes, sintesis.strip(),
                              descartados, time.time() - t0, self.llm.settings.model, ambiguos, aviso,
                              self.llamadas + (1 if len(precedentes) >= 2 else 0))
        yield log(f"\nTiempo total: {resultado.segundos:.1f} s")
        yield Evento("resultado", datos={"resultado": resultado})
