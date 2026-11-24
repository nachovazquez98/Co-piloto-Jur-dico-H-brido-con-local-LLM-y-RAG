"""Análisis de errores: un caso por fallo, a partir de una corrida guardada.

`python -m copiloto_juridico.evaluacion errores --corrida <id> [<id> ...]` lee la corrida (regresión,
qc-diagnostico, recuperación, qc o generación), toma de ella qué falló y reconstruye cada caso sin LLM con el código
actual (clasificación, candidatos, metadatos procesales y fichas precalculadas). Si el commit o el corpus no son los
de la corrida, el reporte lo advierte. Cada fallo se clasifica en una taxonomía fija, en el orden del pipeline
(el primer paso que falla es la causa; los demás se anotan como "también"):

- `cobertura`: no hay sentencias de ese tipo en el corpus;
- `clasificacion`: la familia correcta quedó fuera del top 3;
- `orden`: la familia estaba, pero el precedente relevante quedó fuera del top k;
- `qc1_descarte`: QC_1 eliminó un candidato relevante;
- `ficha`: el sentido del fallo, el delito o los artículos de la ficha no coinciden;
- `otro`: la corrida marca un fallo que no encaja en lo anterior.

Sin juicios humanos, "relevante" = de la familia correcta (lo dice el reporte). Salida: `evaluacion/errores/<id>.md`
y, con todas las corridas dadas, `evaluacion/errores/casos_presentacion.md` (un caso por tipo, máximo 4).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import config

ERRORES_DIR = config.EVAL_DIR / "errores"
TIPOS = {
    "cobertura": "no hay sentencias de ese tipo en el corpus",
    "clasificacion": "la familia correcta quedó fuera del top 3",
    "orden": "la familia estaba, pero el precedente relevante quedó fuera del top k",
    "qc1_descarte": "QC_1 eliminó un candidato relevante",
    "ficha": "el sentido del fallo, el delito o los artículos de la ficha no coinciden",
    "otro": "la corrida marca un fallo que no encaja en los tipos anteriores",
}


@dataclass
class Devuelto:
    doc_id: str
    toca: str
    familia: str
    acto: str
    relevante: bool | None
    qc1: str | None = None              # "SI" / "NO" / "—" si la corrida lo midió
    verificar: list[str] = field(default_factory=list)


@dataclass
class Caso:
    id: str
    consulta: str
    familia_esperada: str | None        # None = sin tipo en el corpus
    top3: list[tuple[str, float]] = field(default_factory=list)
    devueltos: list[Devuelto] = field(default_factory=list)   # los k primeros candidatos (antes de QC_1)
    descartes_qc1: list[Devuelto] = field(default_factory=list)
    acto_consulta: str | None = None
    en_corpus: int = 0                  # sentencias de la familia esperada en el corpus
    marca_corrida: str | None = None    # fallo que reporta la corrida (si lo dice explícitamente)
    nota: str = ""
    tipo: str = ""
    tambien: list[str] = field(default_factory=list)


def tipos_presentes(c: Caso, k: int) -> list[str]:
    """Tipos de fallo del caso, en el orden del pipeline. Vacío = sin fallo."""
    out = []
    if c.familia_esperada is None:
        out.append("cobertura")
    elif c.familia_esperada not in [f for f, _ in c.top3]:
        out.append("clasificacion")
    else:
        relevantes = sum(bool(d.relevante) for d in c.devueltos[:k])
        if relevantes < min(k, c.en_corpus or k):
            out.append("orden")
    if any(d.relevante and d.qc1 == "NO" for d in c.devueltos + c.descartes_qc1):
        out.append("qc1_descarte")
    if any(d.verificar for d in c.devueltos):
        out.append("ficha")
    if not out and c.marca_corrida:
        out.append("otro")
    return out


def clasificar(c: Caso, k: int) -> Caso:
    tipos = tipos_presentes(c, k)
    if c.marca_corrida in TIPOS:  # lo que la corrida midió manda; la reconstrucción solo agrega
        tipos = [c.marca_corrida] + [t for t in tipos if t not in (c.marca_corrida, "otro")]
    c.tipo, c.tambien = (tipos[0], tipos[1:]) if tipos else ("", [])
    return c


# --------------------------------------------------------------------------------------------------------------
# Lectura de corridas y reconstrucción (sin LLM)
# --------------------------------------------------------------------------------------------------------------


def cargar_corrida(ident: str, directorio: Path = config.RESULTADOS_DIR) -> tuple[str, dict]:
    """La corrida por su nombre de archivo (con o sin .json) o un prefijo único."""
    nombre = ident.removesuffix(".json").removesuffix(".md")
    candidatos = sorted(p for p in directorio.glob("*.json") if p.stem == nombre) or \
        sorted(p for p in directorio.glob(f"{nombre}*.json"))
    if len(candidatos) != 1:
        raise FileNotFoundError(f"La corrida «{ident}» {'no existe' if not candidatos else 'es ambigua'} en "
                                f"{directorio}.")
    return candidatos[0].stem, json.loads(candidatos[0].read_text(encoding="utf-8"))


def filas(corrida: dict, titulo: str) -> list[dict]:
    t = corrida.get("tablas", {}).get(titulo)
    return t["filas"] if t else []


def _sin_toca(doc_id: str) -> str:
    """Sin número de toca legible en el encabezado: la fecha y hora de publicación del nombre del archivo."""
    return f"[{doc_id.split('_')[0]}]"


class Reconstructor:
    """Clasificación, candidatos y marcas de ficha de una consulta con el código actual (igual que la app)."""

    def __init__(self, store, clasificador):
        from ..fichas_precalculadas import cargar as cargar_fichas
        from ..metadatos import cargar as cargar_metadatos
        from .familias import cargar_familias

        self.store, self.clasificador = store, clasificador
        corpus = store.corpus
        self.familias = cargar_familias(corpus["tipo"].unique().tolist())
        self.actos = {d: m["acto_impugnado"] for d, m in cargar_metadatos().items()}
        self.por_familia = Counter(self.familias.get(t, t) for t in corpus.loc[corpus["materia"] == "penal", "tipo"])
        self.fichas = {r["doc_id"]: r for r in cargar_fichas().to_dict("records")}

    def marcas_ficha(self, doc_id: str, tipo: str) -> list[str]:
        """Marcas "verificar" de la ficha precalculada + delito de la ficha de otra familia que la sentencia."""
        from ..fichas import FichaPenal, datos_penales, familia_del_delito
        from .familias import FAMILIA_OTROS, etiqueta_familia

        r = self.fichas.get(doc_id)
        if r is None:
            return []
        if not r["json_valido"]:
            return [f"ficha no generada: {r['error'] or 'JSON inválido'}"]
        marcas = list(json.loads(r["verificar"] or "[]"))
        delito = datos_penales(FichaPenal.desde_guardada(r["ficha"]).campos_markdown()).get("delito") or ""
        fam, fam_doc = familia_del_delito(delito), self.familias.get(tipo, tipo)
        if fam not in (FAMILIA_OTROS, fam_doc):
            marcas.append(f"delito: la ficha dice {etiqueta_familia(fam)}, la sentencia es de {etiqueta_familia(fam_doc)}")
        return marcas

    def devuelto_por_id(self, doc_id: str, relevante: bool | None, qc1: str | None = None) -> Devuelto:
        from ..metadatos import toca_de

        if doc_id not in self.store.corpus.index:
            return Devuelto(doc_id, "—", "—", "—", relevante, qc1)
        row = self.store.corpus.loc[doc_id]
        return Devuelto(doc_id, toca_de(row["texto"]) or _sin_toca(doc_id), self.familias.get(row["tipo"], row["tipo"]),
                        self.actos.get(doc_id, "—"), relevante, qc1)

    def caso(self, ident: str, texto: str, conceptos: str, familia_esperada: str | None, k: int,
             relevancia: dict[str, int] | None = None, qc1: dict[str, str] | None = None) -> Caso:
        from ..embeddings import embed_query
        from ..metadatos import toca_de
        from ..pipeline import buscar_candidatos
        from .experimentos import familia_de_categoria

        b = buscar_candidatos(self.store, self.clasificador, texto, conceptos, embed_query(texto), k)
        tabla = b.clasificacion.tabla.head(3)
        top3 = [(familia_de_categoria(c, self.familias), float(s)) for c, s in zip(tabla["tipo"], tabla["score_híbrido"])]

        def devuelto(cand) -> Devuelto:
            fam = self.familias.get(cand.tipo, cand.tipo)
            if relevancia:
                rel = None if cand.doc_id not in relevancia else relevancia[cand.doc_id] >= 1
            else:
                rel = None if familia_esperada is None else fam == familia_esperada
            return Devuelto(cand.doc_id, toca_de(cand.texto) or _sin_toca(cand.doc_id), fam, self.actos.get(cand.doc_id, "—"), rel,
                            (qc1 or {}).get(cand.doc_id), self.marcas_ficha(cand.doc_id, cand.tipo))

        todos = [devuelto(c) for c, _ in b.candidatos]
        # los veredictos de QC_1 de la corrida pueden ir más allá de los k primeros (qc-diagnostico usa 5)
        descartes = [d for d in todos[k:] if d.qc1 == "NO"]
        return Caso(ident, texto, familia_esperada, top3, todos[:k], descartes, b.acto_consulta,
                    self.por_familia.get(familia_esperada, 0) if familia_esperada else 0)


def _avisos(corrida: dict) -> list[str]:
    from .reporte import info_commit

    out = []
    actual = info_commit()
    if corrida.get("commit") != actual["commit"] or corrida.get("sucio"):
        out.append(f"La corrida es del commit `{corrida.get('commit')}`; los casos se reconstruyen con el código actual "
                   f"(`{actual['commit']}`{', con cambios sin confirmar' if actual['sucio'] else ''}).")
    try:
        from ..datos.indice import MANIFEST

        huella = json.loads((config.INDICE_DIR / MANIFEST).read_text()).get("corpus_fingerprint")
    except (OSError, ValueError):
        huella = None
    if huella and corrida.get("corpus_fingerprint") and huella != corrida["corpus_fingerprint"]:
        out.append(f"El corpus cambió desde la corrida ({corrida['corpus_fingerprint']} → {huella}).")
    return out


def casos_de_ejemplos(rec: Reconstructor, k: int, qc1: dict[str, dict[str, str]] | None = None) -> list[Caso]:
    """Los 16 ejemplos penales de la app, con los conceptos precalculados de la regresión."""
    from .regresion import EJEMPLOS_PENALES, conceptos_para, leer_conceptos

    cache = leer_conceptos()
    return [rec.caso(e.etiqueta, e.texto, conceptos_para(e.texto, "v11", cache),
                     None if e.sin_tipo_en_corpus else rec.familias[e.tipo_corpus], k,
                     qc1=(qc1 or {}).get(e.etiqueta)) for e in EJEMPLOS_PENALES]


def _consultas_de(corrida: dict, rec: Reconstructor):
    from .conjuntos import cargar_juicios, cargar_todas, relevancia_por_par
    from .experimentos import conceptos_de, leer_conceptos
    from .referencias import CONSENSO, JUICIOS_AUTO_PATH

    particion = corrida.get("parametros", {}).get("particion")
    todas = cargar_todas(config.CONSULTAS_PATH, corpus=rec.store.corpus, familias=rec.familias, permitir_vacias=True)
    consultas = [c for c in todas if c.consulta.strip() and (particion in (None, "todas") or c.particion == particion)]
    conceptos, _ = conceptos_de(consultas, leer_conceptos())
    tabla = cargar_juicios(JUICIOS_AUTO_PATH, {c.id for c in todas})
    juicios = relevancia_por_par(tabla[tabla["anotador"] == CONSENSO])
    return consultas, conceptos, juicios


def analizar(ident: str, store, clasificador) -> tuple[str, dict, list[Caso], list[dict], list[str]]:
    """(nombre, corrida, casos con fallo, casos corregidos, avisos) de una corrida guardada."""
    nombre, corrida = cargar_corrida(ident)
    comando = corrida["comando"]
    rec = Reconstructor(store, clasificador)
    k = config.RAG_NUM_RESULTS_DESIRED
    avisos = _avisos(corrida)
    corregidos: list[dict] = []

    if comando == "regresion":
        detalle = {f["ejemplo"]: f for f in filas(corrida, "Detalle por ejemplo")
                   if f["conjunto"] == "penal (16)" and f["variante"].startswith("Configuración actual")}
        orden = {f["ejemplo"]: f for f in filas(corrida, "Candidatos por ejemplo")}
        casos = casos_de_ejemplos(rec, k)
        for c in casos:
            if detalle.get(c.id, {}).get("acierto") == "no":
                c.marca_corrida = "clasificacion"
            o = orden.get(c.id)
            if o and o.get("aciertos facetas") is not None and o["aciertos facetas"] < k \
                    and c.marca_corrida is None:
                c.marca_corrida = "orden"
            if o and o.get("aciertos rrf") is not None and o["aciertos facetas"] > o["aciertos rrf"]:
                corregidos.append({"caso": c.id, "acto de la consulta": o["acto consulta"],
                                   "antes (orden RRF)": o["top-3 rrf"], "después (familia + acto)": o["top-3 facetas"],
                                   "aciertos": f"{o['aciertos rrf']:.0f}/{k} → {o['aciertos facetas']:.0f}/{k}"})
    elif comando == "qc-diagnostico":
        det = filas(corrida, "Veredictos de QC_1 por candidato (configuración de la app)")
        if not det:
            raise ValueError(f"{nombre} no guarda los veredictos por candidato: vuelve a correr `qc-diagnostico`.")
        qc1: dict[str, dict[str, str]] = {}
        for f in det:
            qc1.setdefault(f["ejemplo"], {})[f["doc_id"]] = f["veredicto"]
        k_qc = int(corrida.get("parametros", {}).get("k", 5))
        casos = casos_de_ejemplos(rec, k_qc, qc1)
        for c in casos:  # solo los descartes; el resto del caso es contexto
            c.marca_corrida = "qc1_descarte" if any(d.relevante and d.qc1 == "NO" for d in c.devueltos) else None
        casos = [c for c in casos if c.marca_corrida]
        for c in casos:
            c.devueltos, c.descartes_qc1 = c.devueltos[:k], [d for d in c.devueltos[k:] if d.qc1 == "NO"]
            c.nota = f"QC_1 se evaluó sobre los {k_qc} primeros candidatos; se muestran los {k} primeros."
    elif comando in ("recuperacion", "qc"):
        consultas, conceptos, juicios = _consultas_de(corrida, rec)
        qc1: dict[str, dict[str, str]] = {}
        if comando == "qc":
            for f in filas(corrida, "Veredictos de QC_1 por par"):
                if f["variante"] == config.QC1_CONTEXTO:
                    qc1.setdefault(f["consulta"], {})[f["doc_id"]] = f["veredicto"]
        errores_cls = {f["consulta"] for f in filas(corrida, "Errores de clasificación")}
        casos = []
        for c in consultas:
            if comando == "qc" and c.id not in qc1:
                continue
            caso = rec.caso(c.id, c.consulta, conceptos[c.id], c.familia, k, juicios.get(c.id), qc1.get(c.id))
            if c.id in errores_cls:
                caso.marca_corrida = "clasificacion"
            if comando == "qc":  # descartes medidos por la corrida, sobre todos los pares juzgados
                caso.descartes_qc1 = [rec.devuelto_por_id(d, True, "NO") for d, v in qc1[c.id].items()
                                      if v == "NO" and juicios.get(c.id, {}).get(d, 0) >= 1]
                caso.marca_corrida = "qc1_descarte" if caso.descartes_qc1 else None
            casos.append(caso)
        if comando == "qc":
            casos = [c for c in casos if c.marca_corrida]
    elif comando == "fichas-referencia":
        casos = []
        for f in filas(corrida, "Fichas que no coinciden"):
            motivos = [m for m, mal in (
                (f"sentido: ficha {f.get('sentido_ficha')}, referencia {f.get('sentido_ref')}",
                 f.get("sentido_ficha") != f.get("sentido_ref")),
                (f"familia: ficha «{f.get('familia_ficha')}», referencia «{f.get('familia_ref')}»",
                 f.get("familia_ficha") != f.get("familia_ref")),
            ) if mal]
            if not motivos:
                continue
            d = rec.devuelto_por_id(f["doc_id"], None)
            d.verificar = motivos
            casos.append(Caso(f["doc_id"], "(ficha contra la referencia automática; sin consulta)", None,
                              devueltos=[d], marca_corrida="ficha"))
        for c in casos:
            c.tipo, c.tambien = c.marca_corrida, []
        return nombre, corrida, casos, corregidos, avisos
    else:
        raise ValueError(f"`errores` no sabe leer corridas de `{comando}` (sí: regresion, qc-diagnostico, "
                         "recuperacion, qc, fichas-referencia).")
    casos = [clasificar(c, k) for c in casos]
    return nombre, corrida, [c for c in casos if c.tipo], corregidos, avisos


# --------------------------------------------------------------------------------------------------------------
# Reportes
# --------------------------------------------------------------------------------------------------------------


def _etq(f: str | None) -> str:
    from .familias import etiqueta_familia

    return etiqueta_familia(f) if f else "—"


def _si_no(v: bool | None) -> str:
    return "—" if v is None else ("sí" if v else "no")


def que_paso(c: Caso, k: int) -> str:
    """Una frase por caso, para la diapositiva."""
    fams = " · ".join(_etq(d.familia) for d in c.devueltos[:k])
    if c.tipo == "cobertura":
        return f"No hay sentencias de ese tipo en el corpus; los precedentes más cercanos son de {fams}."
    if c.tipo == "clasificacion":
        return (f"La familia correcta ({_etq(c.familia_esperada)}) no quedó en el top 3 "
                f"({' · '.join(_etq(f) for f, _ in c.top3)}).")
    if c.tipo == "orden":
        rel = sum(bool(d.relevante) for d in c.devueltos[:k])
        return (f"{_etq(c.familia_esperada)} quedó en el top 3 de familias, pero solo {rel} de los {k} primeros "
                f"precedentes {'es' if rel == 1 else 'son'} de esa familia ({fams}).")
    if c.tipo == "qc1_descarte":
        d = next(d for d in c.devueltos + c.descartes_qc1 if d.relevante and d.qc1 == "NO")
        return f"QC_1 descartó el toca {d.toca} ({_etq(d.familia)}, {d.acto}), que sí es de la familia de la consulta."
    if c.tipo == "ficha":
        d = next(d for d in c.devueltos if d.verificar)
        return f"Ficha del toca {d.toca}: {d.verificar[0]}."
    return c.nota or "La corrida marca un fallo que la reconstrucción no explica."


def markdown_corrida(nombre: str, corrida: dict, casos: list[Caso], corregidos: list[dict], avisos: list[str]) -> str:
    import pandas as pd

    from .reporte import tabla_markdown

    k = config.RAG_NUM_RESULTS_DESIRED
    conteo = Counter(c.tipo for c in casos)
    lineas = [f"# Errores de `{nombre}`", "",
              f"- Comando: `{corrida['comando']}` · commit `{corrida.get('commit')}` · corpus `{corrida.get('corpus_fingerprint')}`",
              f"- {'Diagnóstico (consultas de ejemplo, no evaluación).' if corrida['comando'] in ('regresion', 'qc-diagnostico') else 'Partición: ' + str(corrida.get('parametros', {}).get('particion'))}",
              "- Relevante = " + ("relevancia ≥ 1 en la referencia automática." if corrida["comando"] in
                                 ("recuperacion", "qc") else "de la familia correcta."),
              "- Fichas: precalculadas (`data/fichas/fichas-penal.parquet`); las marcas son las deterministas de la app.",
              ""]
    lineas += [f"> ⚠️ {a}" for a in avisos] + ([""] if avisos else [])
    lineas += ["## Conteo por tipo", "", tabla_markdown(pd.DataFrame(
        [{"tipo": t, "fallos": conteo.get(t, 0), "qué es": d} for t, d in TIPOS.items()])), ""]
    for i, c in enumerate(casos, 1):
        lineas += [f"## {i}. {c.id} — `{c.tipo}`" + (f" (también: {', '.join(c.tambien)})" if c.tambien else ""), "",
                   "> " + c.consulta.replace("\n", " ")[:600], "",
                   f"- Qué pasó: {que_paso(c, k)}",
                   f"- Familia esperada: {_etq(c.familia_esperada) if c.familia_esperada else 'sin tipo en el corpus'}"
                   + (f" ({c.en_corpus} sentencias en el corpus)" if c.familia_esperada else ""),
                   "- Top 3 de familias: " + (" · ".join(f"{_etq(f)} ({s:.3f})" for f, s in c.top3) or "—"),
                   f"- Acto impugnado de la consulta: {c.acto_consulta or 'no lo menciona'}"]
        if c.nota:
            lineas.append(f"- Nota: {c.nota}")
        lineas += ["", tabla_markdown(pd.DataFrame([{
            "#": j, "toca": d.toca, "familia": _etq(d.familia), "acto impugnado": d.acto, "relevante": _si_no(d.relevante),
            "QC_1": d.qc1 or "—", "ficha: verificar": "; ".join(d.verificar) or "—"}
            for j, d in enumerate(c.devueltos, 1)])), ""]
        if c.descartes_qc1:
            lineas += ["Descartes de QC_1 fuera de los primeros: " + "; ".join(
                f"toca {d.toca} ({_etq(d.familia)}, relevante: {_si_no(d.relevante)})" for d in c.descartes_qc1), ""]
    if corregidos:
        lineas += ["## Corregidos por el orden por facetas (antes → después)", "",
                   "Casos que con el orden RRF tenían menos precedentes de la familia correcta que con el orden por "
                   "familia + acto impugnado (`ORDEN_CANDIDATOS`). Con RRF eran fallos de tipo `orden`.", "",
                   tabla_markdown(pd.DataFrame(corregidos)), ""]
    return "\n".join(lineas)


def seleccionar(casos: list[tuple[str, Caso]], n: int = 4) -> list[tuple[str, Caso]]:
    """Un caso por tipo presente (los tipos más frecuentes primero, máximo n); dentro de cada tipo, el más claro
    (el que tiene menos precedentes relevantes entre los devueltos)."""
    conteo = Counter(c.tipo for _, c in casos)
    out = []
    for tipo, _ in sorted(conteo.items(), key=lambda kv: (-kv[1], list(TIPOS).index(kv[0])))[:n]:
        del_tipo = [x for x in casos if x[1].tipo == tipo]
        out.append(min(del_tipo, key=lambda x: sum(bool(d.relevante) for d in x[1].devueltos)))
    return out


def markdown_presentacion(elegidos: list[tuple[str, Caso]], corregidos: list[dict]) -> str:
    import pandas as pd

    from .reporte import tabla_markdown

    k = config.RAG_NUM_RESULTS_DESIRED

    def corta(t: str, n: int = 18) -> str:
        p = t.split()
        return " ".join(p[:n]) + ("…" if len(p) > n else "")

    lineas = ["# Casos de error para la presentación", "",
              "Elegidos automáticamente por `errores` (uno por tipo de error presente). Diagnóstico con las consultas "
              "de ejemplo: no son tasas de error del sistema.", "",
              tabla_markdown(pd.DataFrame([{"tipo": c.tipo, "consulta": corta(c.consulta), "qué pasó": que_paso(c, k),
                                            "corrida": n} for n, c in elegidos])), ""]
    if corregidos:
        lineas += ["Corregidos en v11.1 por el orden por familia + acto impugnado (antes eran fallos de `orden`):", "",
                   tabla_markdown(pd.DataFrame([{"caso": f["caso"], "antes": f["antes (orden RRF)"],
                                                 "después": f["después (familia + acto)"], "de la familia correcta":
                                                 f["aciertos"]} for f in corregidos])), ""]
    return "\n".join(lineas)


def ejecutar(idents: list[str], store, clasificador, directorio: Path = ERRORES_DIR, log=print) -> list[Path]:
    directorio.mkdir(parents=True, exist_ok=True)
    escritos, todos, corregidos_todos = [], [], []
    for ident in idents:
        nombre, corrida, casos, corregidos, avisos = analizar(ident, store, clasificador)
        p = directorio / f"{nombre}.md"
        p.write_text(markdown_corrida(nombre, corrida, casos, corregidos, avisos), encoding="utf-8")
        log(f"{nombre}: {len(casos)} fallos · " + ", ".join(f"{t} {n}" for t, n in Counter(c.tipo for c in casos).items())
            + f" → {p}")
        escritos.append(p)
        todos += [(nombre, c) for c in casos]
        corregidos_todos += corregidos
    p = directorio / "casos_presentacion.md"
    p.write_text(markdown_presentacion(seleccionar(todos), corregidos_todos), encoding="utf-8")
    escritos.append(p)
    return escritos
