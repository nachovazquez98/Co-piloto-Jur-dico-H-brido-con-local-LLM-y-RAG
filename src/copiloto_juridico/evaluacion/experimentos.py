"""Experimentos de evaluación (sección 6 del informe).

Diseño:
- Los experimentos de **recuperación y clasificación** no llaman al LLM: los conceptos clave de cada consulta se
  precalculan una vez (`conceptos`) y se leen de caché. Así son rápidos, baratos y reproducibles.
- Los de **QC, generación y latencia** sí usan el LLM. Las respuestas se guardan en una caché en disco
  (`evaluacion/cache/llm.jsonl`) para que una corrida interrumpida continúe donde quedó; latencia nunca usa
  caché porque mide tiempos reales.
- Toda función que calcula métricas recibe datos ya preparados, para poder probarla sin GPU ni Ollama.

Advertencia metodológica (*leave-one-out*): en el conjunto A la consulta se redactó a partir de una sentencia
que está en el corpus. Esa sentencia suele recuperarse y su tipo "vota" por sí mismo en el clasificador, lo que
infla la exactitud de clasificación respecto al uso real (donde el caso del usuario no está en el corpus). Por
eso la clasificación se reporta también *sin la fuente* (la sentencia fuente se excluye de la búsqueda que
alimenta el voto y del k-NN). La recuperación del documento fuente, en cambio, sí debe incluirla.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .. import config, prompts
from ..clasificador import COMPONENTES, PENALIZACIONES, PesosClasificador, puntuar
from ..fichas import (coincide_texto, datos_penales, extraer_articulos, f1_conjuntos, normalizar,
                      normalizar_sentido, parsear_campos)
from . import metricas as m
from .conjuntos import Consulta
from .reporte import Corrida, fmt_ic

# --------------------------------------------------------------------------------------------------------------
# Utilidades de métricas por consulta
# --------------------------------------------------------------------------------------------------------------


def familia_de_categoria(categoria: str, familias: dict[str, str]) -> str:
    """Familia de una categoría del clasificador: a nivel familia la clave ya la contiene."""
    if categoria.startswith("penal/familia/"):
        return categoria.split("/", 2)[2]
    return familias.get(categoria, categoria)


def resumen_categorias(verdad_tipo: str, verdad_familia: str, ranking: list[str],
                       tipos_de: Callable[[str], list[str]], familias: dict[str, str], k: int = 3) -> dict:
    """Aciertos de clasificación de una consulta.

    - hit3_tipo: el tipo correcto queda dentro del filtro (los tipos que cubren las k primeras categorías);
    - hit3_familia: alguna de las k primeras categorías es de la familia correcta;
    - top1_tipo / top1_familia: la primera categoría es la correcta (top1_tipo es NaN si la categoría agrupa
      varios tipos, como al clasificar por familia).
    """
    top_k = [t for c in ranking[:k] for t in tipos_de(c)]
    fams_k = {familia_de_categoria(c, familias) for c in ranking[:k]}
    primera = ranking[0] if ranking else ""
    tipos_primera = tipos_de(primera) if primera else []
    return {
        "hit3_tipo": float(verdad_tipo in top_k),
        "hit3_familia": float(verdad_familia in fams_k),
        "top1_tipo": float(tipos_primera == [verdad_tipo]) if len(tipos_primera) <= 1 else float("nan"),
        "top1_familia": float(familia_de_categoria(primera, familias) == verdad_familia) if primera else 0.0,
        "pred_tipo": primera, "pred_familia": familia_de_categoria(primera, familias) if primera else "",
    }


def metricas_documento(ids: list[str], fuente: str) -> dict:
    """Recuperación del documento conocido: Hit@1/3/5/10 y rango recíproco (lista de hasta 30)."""
    rels = [1 if d == fuente else 0 for d in ids]
    return {"hit1": m.hit_at_k(rels, 1), "hit3": m.hit_at_k(rels, 3), "hit5": m.hit_at_k(rels, 5),
            "hit10": m.hit_at_k(rels, 10), "rr": m.reciprocal_rank(rels), "rango": m.rango_primer_acierto(rels)}


def relevancias_metadatos(ids: list[str], tipo: str, familia: str, doc_tipo: dict[str, str],
                          familias: dict[str, str]) -> list[int]:
    """Relevancia aproximada por metadatos: 2 = mismo tipo, 1 = misma familia, 0 = otra cosa.

    Es una aproximación barata del juicio humano (no ve el contexto fáctico); sirve mientras no hay juicios y
    para comparar variantes, pero el número reportable de nDCG es el del conjunto B anotado."""
    out = []
    for d in ids:
        t = doc_tipo.get(d)
        out.append(2 if t == tipo else 1 if familias.get(t, t) == familia else 0)
    return out


def ideales_metadatos(tipo: str, familia: str, conteo_tipos: dict[str, int], familias: dict[str, str]) -> list[int]:
    """Relevancias de todos los documentos relevantes del corpus para el orden ideal del nDCG por metadatos."""
    n_tipo = conteo_tipos.get(tipo, 0)
    n_fam = sum(n for t, n in conteo_tipos.items() if familias.get(t, t) == familia) - n_tipo
    return [2] * n_tipo + [1] * max(0, n_fam)


def metricas_graduadas(ids: list[str], juicios: dict[str, int], k: int = 5) -> dict:
    """nDCG@k y P@k con juicios humanos. Los documentos no juzgados cuentan como 0 (supuesto conservador);
    `cobertura` dice qué fracción del top k estaba juzgada, para saber cuánto pesa ese supuesto."""
    rels = [juicios.get(d, 0) for d in ids]
    return {"ndcg5": m.ndcg_at_k(rels, k, list(juicios.values())), "p5": m.precision_at_k(rels, k),
            "cobertura": sum(1 for d in ids[:k] if d in juicios) / k}


def ranking_knn(vecinos: list, n: int = 10) -> list[str]:
    """Línea base k-NN: tipos ordenados por cuántas de las n sentencias más cercanas (búsqueda densa) los
    tienen; empates por similitud acumulada. El primero es el "tipo mayoritario"."""
    cuenta: Counter = Counter()
    sim: dict[str, float] = {}
    for c in vecinos[:n]:
        cuenta[c.tipo] += 1
        sim[c.tipo] = sim.get(c.tipo, 0.0) + c.similitud
    return sorted(cuenta, key=lambda t: (-cuenta[t], -sim[t]))


def ranking_categorias(comp: pd.DataFrame, pesos: PesosClasificador) -> list[str]:
    """Categorías ordenadas por score híbrido (mismo orden estable que usa la app)."""
    if comp.empty:
        return []
    orden = np.argsort(-puntuar(comp, pesos), kind="stable")
    return comp["tipo"].to_numpy()[orden].tolist()


def _fila_proporcion(nombre_col: str, valores: list[float]) -> dict:
    x = [v for v in valores if not (isinstance(v, float) and np.isnan(v))]
    if not x:
        return {nombre_col: float("nan"), f"{nombre_col} IC95": "—"}
    r = m.proporcion_con_ic(x)
    return {nombre_col: r["valor"], f"{nombre_col} IC95": f"[{r['ic_inf']:.3f}, {r['ic_sup']:.3f}]"}


def _fila_promedio(nombre_col: str, valores: list[float]) -> dict:
    if not valores:
        return {nombre_col: float("nan"), f"{nombre_col} IC95": "—"}
    r = m.promedio_con_ic(valores, semilla=config.EVAL_SEED)
    return {nombre_col: r["valor"], f"{nombre_col} IC95": f"[{r['ic_inf']:.3f}, {r['ic_sup']:.3f}]"}


def _fila_diferencia(a: list[float], b: list[float]) -> dict:
    d = m.bootstrap_diferencia_pareada(a, b, config.EVAL_BOOTSTRAP, config.EVAL_SEED)
    return {"Δ": d["diferencia"], "Δ IC95": f"[{d['ic_inf']:+.3f}, {d['ic_sup']:+.3f}]",
            "¿significativa?": "sí" if d["significativa"] else "no", "gana A / B / empate":
            f"{d['gana_a']} / {d['gana_b']} / {d['empates']}"}


def auc_roc(positivos: list[float], negativos: list[float]) -> float:
    """Área bajo la curva ROC = probabilidad de que un positivo tenga puntaje mayor que un negativo
    (estadístico de Mann-Whitney; empates cuentan 1/2). 0.5 = azar, 1 = separación perfecta."""
    if not positivos or not negativos:
        return float("nan")
    gana = sum((p > n) + 0.5 * (p == n) for p in positivos for n in negativos)
    return gana / (len(positivos) * len(negativos))


# --------------------------------------------------------------------------------------------------------------
# Malla de pesos
# --------------------------------------------------------------------------------------------------------------


def malla_pesos(paso: float = 0.05, n: int = len(COMPONENTES)) -> np.ndarray:
    """Todas las combinaciones de n pesos ≥ 0 múltiplos de `paso` que suman 1 (con paso 0.05 y 5 pesos:
    C(24, 4) = 10,626 combinaciones)."""
    partes = int(round(1 / paso))
    filas = []
    for cortes in itertools.combinations(range(partes + n - 1), n - 1):
        prev, fila = -1, []
        for c in cortes:
            fila.append(c - prev - 1)
            prev = c
        fila.append(partes + n - 2 - prev)
        filas.append(fila)
    return np.array(filas, dtype=float) / partes


@dataclass
class CasoMalla:
    """Lo que la malla necesita de una consulta: señales de los candidatos y qué candidato es correcto."""

    F: np.ndarray            # (candidatos × 5) señales en el orden de COMPONENTES
    penal: np.ndarray        # (candidatos,) producto de penalizaciones activas
    es_tipo: np.ndarray      # (candidatos,) bool: la categoría es el tipo correcto
    es_familia: np.ndarray   # (candidatos,) bool: la categoría es de la familia correcta


def caso_malla(comp: pd.DataFrame, verdad_tipo: str, verdad_familia: str, familias: dict[str, str],
               pesos_base: PesosClasificador) -> CasoMalla:
    penal = np.ones(len(comp))
    if pesos_base.penal_longitud:
        penal *= comp["penal_longitud"].to_numpy(dtype=float)
    if pesos_base.penal_tokens:
        penal *= comp["penal_tokens"].to_numpy(dtype=float)
    tipos = comp["tipo"].tolist()
    return CasoMalla(comp[list(COMPONENTES)].to_numpy(dtype=float), penal,
                     np.array([t == verdad_tipo for t in tipos]),
                     np.array([familia_de_categoria(t, familias) == verdad_familia for t in tipos]))


def evaluar_malla(casos: list[CasoMalla], W: np.ndarray, k: int = 3) -> dict[str, np.ndarray]:
    """Para cada combinación de pesos (filas de W), el promedio entre consultas de Hit@k (tipo y familia) y
    top-1. Vectorizado: score = (F · wᵀ) × penalización, orden estable como en la app."""
    tot = {"hit3_tipo": np.zeros(len(W)), "hit3_familia": np.zeros(len(W)), "top1_tipo": np.zeros(len(W)),
           "top1_familia": np.zeros(len(W))}
    for c in casos:
        if len(c.F) == 0:
            continue
        S = (c.F @ W.T) * c.penal[:, None]                    # candidatos × combinaciones
        orden = np.argsort(-S, axis=0, kind="stable")[:k]    # k × combinaciones
        tot["hit3_tipo"] += c.es_tipo[orden].any(axis=0)
        tot["hit3_familia"] += c.es_familia[orden].any(axis=0)
        tot["top1_tipo"] += c.es_tipo[orden[0]]
        tot["top1_familia"] += c.es_familia[orden[0]]
    n = max(1, len(casos))
    return {k_: v / n for k_, v in tot.items()}


# --------------------------------------------------------------------------------------------------------------
# Caché de LLM y contador de llamadas
# --------------------------------------------------------------------------------------------------------------


class LLMConCache:
    """Envuelve un cliente de LLM: guarda cada respuesta en un JSONL indexado por el hash de (modelo,
    parámetros, prompt). Con temperatura 0 y semilla fija, repetir la misma llamada daría lo mismo; la caché
    evita pagar horas de GPU otra vez y permite reanudar una corrida."""

    def __init__(self, cliente, path: Path | None = config.EVAL_DIR / "cache" / "llm.jsonl"):
        self.cliente = cliente
        self.settings = cliente.settings
        self.path = path
        self.memoria: dict[str, str] = {}
        self.llamadas = 0
        self.aciertos = 0
        if path and path.exists():
            for linea in path.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    d = json.loads(linea)
                    self.memoria[d["clave"]] = d["respuesta"]

    def _clave(self, prompt: str) -> str:
        s = self.settings
        base = json.dumps([s.provider, s.model, s.temperature, s.max_tokens, s.num_ctx,
                           getattr(s, "seed", None), prompt], ensure_ascii=False)
        return hashlib.sha256(base.encode()).hexdigest()

    def generate(self, prompt: str) -> str:
        clave = self._clave(prompt)
        self.llamadas += 1
        if clave in self.memoria:
            self.aciertos += 1
            return self.memoria[clave]
        resp = self.cliente.generate(prompt)
        self.memoria[clave] = resp
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"clave": clave, "respuesta": resp}, ensure_ascii=False) + "\n")
        return resp

    def stream(self, prompt: str):
        yield self.generate(prompt)


class LLMContador:
    """Cuenta llamadas al LLM sin alterar nada (para la métrica de llamadas por consulta)."""

    def __init__(self, cliente):
        self.cliente = cliente
        self.settings = cliente.settings
        self.llamadas = 0

    def generate(self, prompt: str) -> str:
        self.llamadas += 1
        return self.cliente.generate(prompt)

    def stream(self, prompt: str):
        self.llamadas += 1
        yield from self.cliente.stream(prompt)


# --------------------------------------------------------------------------------------------------------------
# Entorno (carga de la base, el clasificador y los datos de evaluación)
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Preparada:
    consulta: Consulta
    q: np.ndarray
    conceptos: str
    lexica: str


@dataclass
class Entorno:
    store: object
    clasificador: object
    familias: dict[str, str]
    consultas: list[Consulta]
    conceptos: dict[str, str]
    juicios: dict[str, dict[str, int]]
    corrida: Corrida
    preparadas: list[Preparada] = field(default_factory=list)

    @property
    def doc_tipo(self) -> dict[str, str]:
        return self.store.corpus.set_index("id")["tipo"].to_dict()

    @property
    def conteo_tipos(self) -> dict[str, int]:
        return self.store.corpus["tipo"].value_counts().to_dict()


def clave_conceptos(c: Consulta) -> str:
    """Clave de caché: texto de la consulta + texto del prompt de conceptos (si cambia cualquiera, se recalcula)."""
    return hashlib.sha256(f"{prompts.prompt_conceptos()}|{c.consulta}".encode()).hexdigest()[:16]


def leer_conceptos(path: Path = config.CONCEPTOS_CACHE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def conceptos_de(consultas: list[Consulta], cache: dict) -> tuple[dict[str, str], list[str]]:
    """Conceptos clave en caché por consulta; si faltan (o la consulta cambió), se usa la consulta completa
    (lo mismo que hace la app cuando el LLM falla) y se devuelve la lista de ids sin conceptos."""
    out, faltan = {}, []
    for c in consultas:
        d = cache.get(c.id)
        if d and d.get("clave") == clave_conceptos(c):
            out[c.id] = d["conceptos"]
        else:
            out[c.id] = c.consulta
            faltan.append(c.id)
    return out, faltan


def cargar_entorno(comando: str, particion: str | None, ruta_consultas: Path = config.CONSULTAS_PATH,
                   parametros: dict | None = None, log: Callable[[str], None] = print,
                   con_store: bool = True) -> Entorno:
    """Carga consultas (validadas contra el corpus), familias, juicios, conceptos en caché, la base y el
    clasificador. Marca la corrida como borrador si alguna consulta lo es."""
    from ..datos.corpus import load_corpus
    from .conjuntos import cargar_consultas, cargar_juicios, relevancia_por_par, verificar_fuga
    from .familias import cargar_familias

    corrida = Corrida(comando, parametros={"particion": particion, **(parametros or {})})
    store = clasificador = None
    if con_store:
        from ..clasificador import construir_clasificador
        from ..vectorstore import SentenciasStore

        log("Cargando la base de sentencias...")
        store = SentenciasStore(log=lambda _m: None)
        corpus = store.corpus
        log("Cargando el clasificador...")
        clasificador = construir_clasificador(corpus["tipo"].unique().tolist())
    else:
        corpus = load_corpus()
    familias = cargar_familias(corpus["tipo"].unique().tolist())
    todas = cargar_consultas(ruta_consultas, corpus=corpus, familias=familias, permitir_vacias=True)
    vacias = [c.id for c in todas if not c.consulta.strip()]
    if vacias:
        corrida.advertir(f"{len(vacias)} consultas sin redactar se omiten: {', '.join(vacias[:10])}"
                         f"{'...' if len(vacias) > 10 else ''}")
    consultas = [c for c in todas if c.consulta.strip() and (particion in (None, "todas") or c.particion == particion)]
    if any(c.borrador for c in consultas):
        corrida.borrador = True
        corrida.advertir("Hay consultas en borrador: esta corrida NO es reportable.")
    fugas = verificar_fuga(consultas, corpus)
    if fugas:
        corrida.advertir(f"{len(fugas)} consultas comparten más de 6 palabras seguidas con su sentencia fuente "
                         f"(fuga): {', '.join(sorted(fugas)[:10])}. Corre `validar` para verlas.")
    ids = {c.id for c in todas}
    juicios = relevancia_por_par(cargar_juicios(config.JUICIOS_PATH, ids))
    conceptos, faltan = conceptos_de(consultas, leer_conceptos())
    if faltan:
        corrida.advertir(f"{len(faltan)} consultas sin conceptos en caché (se usa la consulta completa). "
                         "Corre `conceptos` primero para reproducir la app.")
    corrida.extra["n_consultas"] = len(consultas)
    corrida.extra["nivel_clasificacion"] = config.NIVEL_CLASIFICACION
    corrida.extra["prompt_conceptos"] = config.PROMPT_CONCEPTOS_VERSION
    corrida.extra["materias_activas"] = list(config.MATERIAS_ACTIVAS)
    return Entorno(store, clasificador, familias, consultas, conceptos, juicios, corrida)


def preparar(env: Entorno) -> list[Preparada]:
    """Embedding de cada consulta y texto léxico (consulta + conceptos), igual que en la app."""
    if env.preparadas:
        return env.preparadas
    from ..embeddings import embed_texts

    if not env.consultas:
        return []
    embs = embed_texts([c.consulta for c in env.consultas], kind="query")
    env.preparadas = [Preparada(c, e, env.conceptos[c.id], f"{c.consulta} {env.conceptos[c.id]}")
                      for c, e in zip(env.consultas, embs)]
    return env.preparadas


# --------------------------------------------------------------------------------------------------------------
# Recuperación y clasificación
# --------------------------------------------------------------------------------------------------------------


def _clasificacion(env: Entorno, p: Preparada, rrf_k: int | None = None, excluir: set | None = None,
                   pesos: PesosClasificador | None = None):
    amplios = env.store.buscar(p.consulta.consulta, query_embedding=p.q, consulta_lexica=p.lexica, rrf_k=rrf_k,
                               excluir=excluir)
    comp, _ = env.clasificador.componentes(p.consulta.consulta, p.conceptos, p.q, recuperados=amplios)
    return amplios, comp, ranking_categorias(comp, pesos or PesosClasificador.desde_config())


def _lista_pipeline(env: Entorno, p: Preparada, ranking: list[str], amplios: list, n_filtro: int,
                    rrf_k: int | None, excluir: set | None = None) -> list[str]:
    """Lista final de candidatos como la arma la app: filtro con las `n_filtro` primeras categorías y, si da
    pocos, se completa con la búsqueda amplia (`amplios`). 0 = sin filtro."""
    from ..pipeline import combinar_candidatos

    if n_filtro == 0:
        return [c.doc_id for c in amplios]
    filtro = [t for c in ranking[:n_filtro] for t in env.clasificador.tipos_de(c)]
    estrictos = env.store.buscar(p.consulta.consulta, tipos=filtro, query_embedding=p.q, consulta_lexica=p.lexica,
                                 rrf_k=rrf_k, excluir=excluir)
    candidatos, _ = combinar_candidatos(estrictos, amplios, config.RAG_NUM_RESULTS_DESIRED)
    return [c.doc_id for c, _ in candidatos][:config.RAG_NUM_DOCS_TO_FETCH]


VARIANTES_RECUPERACION = [
    # (nombre, modo, rrf_k, categorías en el filtro; 0 = sin filtro)
    ("Solo BM25", "lexica", 60, 0),
    ("Solo embeddings", "densa", 60, 0),
    ("RRF k=20, sin filtro", "hibrida", 20, 0),
    ("RRF k=60, sin filtro", "hibrida", 60, 0),
    ("RRF k=100, sin filtro", "hibrida", 100, 0),
    ("RRF k=60 + filtro top-1", "hibrida", 60, 1),
    ("RRF k=60 + filtro top-3 (app)", "hibrida", 60, 3),
    ("RRF k=60 + filtro top-5", "hibrida", 60, 5),
    ("RRF k=20 + filtro top-3", "hibrida", 20, 3),
    ("RRF k=100 + filtro top-3", "hibrida", 100, 3),
]


def _resumen_cls(nombre: str, rs: list[dict]) -> dict:
    fila = {"variante": nombre, "n": len(rs)}
    for col, key in [("Hit@3 tipo", "hit3_tipo"), ("Hit@3 familia", "hit3_familia"),
                     ("Top-1 tipo", "top1_tipo"), ("Top-1 familia", "top1_familia")]:
        fila.update(_fila_proporcion(col, [r[key] for r in rs]))
    por_tipo = rs and not any(r["pred_tipo"].startswith("penal/familia/") for r in rs)
    fila["F1 macro tipo"] = m.f1_macro([r["tipo"] for r in rs], [r["pred_tipo"] for r in rs]) \
        if por_tipo else float("nan")  # a nivel familia la predicción no es un tipo
    fila["F1 macro familia"] = m.f1_macro([r["familia"] for r in rs], [r["pred_familia"] for r in rs]) \
        if rs else float("nan")
    return fila


def experimento_recuperacion(env: Entorno, log: Callable[[str], None] = print) -> Corrida:
    """Líneas base de recuperación, efecto del filtro, k de RRF y clasificador k-NN (sección 6.3).

    Leave-one-out en todo el conjunto A: la sentencia fuente se excluye de la búsqueda que alimenta el voto del
    clasificador y del k-NN, y de las listas con que se calculan nDCG@5 y P@5 por metadatos (que miden "otros
    precedentes del mismo tipo"). Solo Hit@k y MRR buscan la sentencia fuente en la lista final, porque es el
    objetivo de esa métrica. La clasificación con la fuente incluida queda como tabla de diagnóstico."""
    corrida = env.corrida
    preparadas = preparar(env)
    conocidas = [p for p in preparadas if p.consulta.documento_conocido]
    doc_tipo, conteo = env.doc_tipo, env.conteo_tipos
    tipo_unico = lambda t: [t]  # noqa: E731

    # --- Clasificación ---
    hib = "Híbrido (app)"
    res = {k: [] for k in ("hib", "knn", "hib_diag", "knn_diag")}
    errores = []
    log(f"Clasificación: {len(conocidas)} consultas con documento conocido...")
    for p in conocidas:
        c = p.consulta
        sin = {c.doc_id_fuente}
        _, _, rk = _clasificacion(env, p, excluir=sin)
        _, _, rk_diag = _clasificacion(env, p)
        vec = env.store.buscar(c.consulta, query_embedding=p.q, modo="densa", num_docs=10, excluir=sin)
        vec_diag = env.store.buscar(c.consulta, query_embedding=p.q, modo="densa", num_docs=10)
        for clave_, ranking, tipos_de in (("hib", rk, env.clasificador.tipos_de), ("knn", ranking_knn(vec), tipo_unico),
                                          ("hib_diag", rk_diag, env.clasificador.tipos_de),
                                          ("knn_diag", ranking_knn(vec_diag), tipo_unico)):
            r = resumen_categorias(c.tipo, c.familia, ranking, tipos_de, env.familias)
            r.update(id=c.id, tipo=c.tipo, familia=c.familia, top3=ranking[:3])
            res[clave_].append(r)
        if not res["hib"][-1]["hit3_familia"]:
            errores.append({"consulta": c.id, "partición": c.particion, "tipo correcto": c.tipo,
                            "familia": c.familia, "top-3": " · ".join(res["hib"][-1]["top3"])})

    corrida.tabla(
        "Clasificación de la consulta",
        "Hit@3 = la categoría correcta queda dentro del filtro de 3 categorías (si no, ninguna etapa posterior la "
        "recupera). La sentencia de la que se redactó la consulta no participa (leave-one-out). IC95 de Wilson; "
        "F1 macro sobre la predicción top-1. k-NN = tipo mayoritario de las 10 sentencias más cercanas.",
        pd.DataFrame([_resumen_cls(hib, res["hib"]), _resumen_cls("k-NN 10 (línea base)", res["knn"])]))
    corrida.tabla(
        "Diagnóstico: clasificación con la sentencia fuente incluida",
        "Solo diagnóstico (no es la métrica): cuánto se inflaría la exactitud si la sentencia fuente votara por su "
        "propio tipo.", pd.DataFrame([_resumen_cls(f"{hib}, con fuente", res["hib_diag"]),
                                      _resumen_cls("k-NN 10, con fuente", res["knn_diag"])]), diagnostico=True)
    if errores:
        corrida.tabla("Errores de clasificación", "Consultas cuya familia correcta quedó fuera del top-3.",
                      pd.DataFrame(errores))
    if res["hib"]:
        cm = m.matriz_confusion([r["familia"] for r in res["hib"]], [r["pred_familia"] for r in res["hib"]])
        cm = cm.loc[(cm.sum(axis=1) > 0) | (cm.sum(axis=0) > 0)]
        corrida.tabla("Matriz de confusión por familia (top-1)",
                      "Filas = familia correcta, columnas = familia predicha. La diagonal son aciertos.",
                      cm.reset_index())

    # --- Recuperación ---
    log("Recuperación: variantes de búsqueda...")
    res_rec: dict[str, list[dict]] = {}
    for nombre, modo, k, n_filtro in VARIANTES_RECUPERACION:
        filas = []
        for p in preparadas:
            c = p.consulta
            if c.sin_respuesta:
                continue
            sin = {c.doc_id_fuente} if c.documento_conocido else None

            def lista(excluir):
                if modo != "hibrida":
                    return [x.doc_id for x in env.store.buscar(c.consulta, query_embedding=p.q, modo=modo, rrf_k=k,
                                                               consulta_lexica=p.lexica, excluir=excluir)]
                amplios = env.store.buscar(c.consulta, query_embedding=p.q, consulta_lexica=p.lexica, rrf_k=k,
                                           excluir=excluir)
                _, _, rk = _clasificacion(env, p, rrf_k=k, excluir=sin)  # el voto nunca incluye la fuente
                return _lista_pipeline(env, p, rk, amplios, n_filtro, k, excluir)

            fila = {"id": c.id}
            if c.documento_conocido:
                fila.update(metricas_documento(lista(None), c.doc_id_fuente))
                ids_sin = lista(sin)
                rels = relevancias_metadatos(ids_sin, c.tipo, c.familia, doc_tipo, env.familias)
                ideales = ideales_metadatos(c.tipo, c.familia, {**conteo, c.tipo: conteo.get(c.tipo, 1) - 1},
                                            env.familias)
                fila["ndcg5_meta"] = m.ndcg_at_k(rels, 5, ideales)
                fila["p5_meta"] = m.precision_at_k(rels, 5, umbral=2)
            if c.id in env.juicios:
                fila.update(metricas_graduadas(lista(None), env.juicios[c.id]))
            filas.append(fila)
        res_rec[nombre] = filas

    tabla_doc, tabla_juicios = [], []
    for nombre, filas in res_rec.items():
        doc = [f for f in filas if "hit5" in f]
        fila = {"variante": nombre, "n": len(doc)}
        fila.update(_fila_proporcion("Hit@1", [f["hit1"] for f in doc]))
        fila.update(_fila_proporcion("Hit@5", [f["hit5"] for f in doc]))
        fila["Hit@10"] = float(np.mean([f["hit10"] for f in doc])) if doc else float("nan")
        fila.update(_fila_promedio("MRR", [f["rr"] for f in doc]))
        fila.update(_fila_promedio("nDCG@5 (metadatos)", [f["ndcg5_meta"] for f in doc]))
        fila["P@5 mismo tipo"] = float(np.mean([f["p5_meta"] for f in doc])) if doc else float("nan")
        tabla_doc.append(fila)
        jz = [f for f in filas if "ndcg5" in f]
        if jz:
            fj = {"variante": nombre, "n": len(jz)}
            fj.update(_fila_promedio("nDCG@5", [f["ndcg5"] for f in jz]))
            fj.update(_fila_promedio("P@5", [f["p5"] for f in jz]))
            fj["cobertura juicios top-5"] = float(np.mean([f["cobertura"] for f in jz]))
            tabla_juicios.append(fj)
    corrida.tabla(
        "Recuperación (conjunto A)",
        "Hit@k = la sentencia de la que se redactó la consulta aparece en los k primeros; MRR = promedio de 1/rango "
        "(lista de hasta 30). nDCG@5 y P@5 por metadatos (2 = mismo tipo, 1 = misma familia) se calculan sin la "
        "sentencia fuente: miden si aparecen OTROS precedentes del mismo tipo; no sustituyen al juicio humano. "
        "IC95: Wilson para Hit, bootstrap (1,000) para MRR y nDCG.", pd.DataFrame(tabla_doc))
    if tabla_juicios:
        corrida.tabla("Relevancia juzgada por humanos",
                      "nDCG@5 y P@5 (relevante = 1 o 2) con los juicios anotados; los no juzgados cuentan 0.",
                      pd.DataFrame(tabla_juicios))
    else:
        corrida.advertir("Sin juicios de relevancia anotados: nDCG@5 y P@5 humanos quedan por medir.")

    # --- Comparaciones pareadas clave ---
    comparaciones = []

    def comparar(a: str, b: str, metrica: str, etiqueta: str) -> None:
        fa = {f["id"]: f for f in res_rec[a] if metrica in f}
        fb = {f["id"]: f for f in res_rec[b] if metrica in f}
        comunes = sorted(set(fa) & set(fb))
        if comunes:
            comparaciones.append({"A": a, "B": b, "métrica": etiqueta, "n": len(comunes),
                                  **_fila_diferencia([fa[i][metrica] for i in comunes],
                                                     [fb[i][metrica] for i in comunes])})

    for base in ("Solo BM25", "Solo embeddings"):
        comparar("RRF k=60, sin filtro", base, "rr", "RR (documento fuente)")
        comparar("RRF k=60, sin filtro", base, "ndcg5_meta", "nDCG@5 (metadatos)")
        comparar("RRF k=60, sin filtro", base, "ndcg5", "nDCG@5 (juicios)")
    for metrica, etiqueta in (("hit5", "Hit@5 (documento fuente)"), ("ndcg5_meta", "nDCG@5 (metadatos)"),
                              ("ndcg5", "nDCG@5 (juicios)")):
        comparar("RRF k=60 + filtro top-3 (app)", "RRF k=60, sin filtro", metrica, etiqueta)
    a, b = {r["id"]: r for r in res["hib"]}, {r["id"]: r for r in res["knn"]}
    for metrica in ("hit3_tipo", "hit3_familia"):
        comunes = sorted(set(a) & set(b))
        if comunes:
            comparaciones.append({"A": hib, "B": "k-NN 10", "métrica": metrica, "n": len(comunes),
                                  **_fila_diferencia([a[i][metrica] for i in comunes], [b[i][metrica] for i in comunes])})
    if comparaciones:
        corrida.tabla("Comparaciones pareadas",
                      "Δ = media(A − B) sobre las mismas consultas, IC95 bootstrap pareado. Una mejora se acepta solo "
                      "si el intervalo no incluye 0.", pd.DataFrame(comparaciones))

    # --- Consultas sin respuesta ---
    sin_resp = [p for p in preparadas if p.consulta.sin_respuesta]
    if sin_resp and conocidas:
        corrida.tabla("Detección de consultas sin respuesta", *_tabla_sin_respuesta(env, sin_resp, conocidas))
    elif not sin_resp:
        corrida.advertir("No hay consultas sin respuesta redactadas: la detección de 'sin precedentes del mismo "
                         "tipo' queda por medir.")
    return corrida


def senales_sin_respuesta(tabla: pd.DataFrame, n_filtro: int = config.NUM_CATEGORIAS_FILTRO) -> dict[str, float]:
    """Mismas señales que usa la app para el aviso de cobertura (`clasificador.senales_coincidencia`)."""
    from ..clasificador import senales_coincidencia

    return senales_coincidencia(tabla, n_filtro)


def _tabla_sin_respuesta(env: Entorno, sin_resp: list[Preparada], conocidas: list[Preparada]):
    pesos = PesosClasificador.desde_config()

    def senales(p: Preparada) -> dict:
        amplios = env.store.buscar(p.consulta.consulta, query_embedding=p.q, consulta_lexica=p.lexica,
                                   excluir={p.consulta.doc_id_fuente} if p.consulta.doc_id_fuente else None)
        comp, _ = env.clasificador.componentes(p.consulta.consulta, p.conceptos, p.q, recuperados=amplios)
        return senales_sin_respuesta(env.clasificador.tabla_desde_componentes(comp, pesos))

    pos = [senales(p) for p in sin_resp]      # positivos: sin respuesta (señal baja = detectado)
    neg = [senales(p) for p in conocidas]
    filas = []
    for s in ("score_hibrido", "coseno", "kem_max"):
        # Detecta "sin respuesta" cuando la señal es BAJA; AUC con signo invertido.
        auc = auc_roc([-x[s] for x in pos], [-x[s] for x in neg])
        mejor = max(sorted({x[s] for x in pos + neg}),
                    key=lambda u: (np.mean([x[s] < u for x in pos]) + np.mean([x[s] >= u for x in neg])) / 2,
                    default=float("nan"))
        filas.append({"señal": s, "AUC": auc, "umbral sugerido (exactitud balanceada)": mejor,
                      "detección sin respuesta": float(np.mean([x[s] < mejor for x in pos])),
                      "falsa alarma en conocidas": float(np.mean([x[s] < mejor for x in neg])),
                      "n sin respuesta": len(pos), "n conocidas": len(neg)})
    desc = ("¿Qué tan bien separa cada señal las consultas sin tipo en el corpus de las que sí tienen? AUC 0.5 = "
            "azar. El umbral se elige en desarrollo y se configura en COPILOTO_UMBRAL_SIN_PRECEDENTES.")
    return desc, pd.DataFrame(filas)


# --------------------------------------------------------------------------------------------------------------
# Ablación y malla de pesos
# --------------------------------------------------------------------------------------------------------------


def _componentes_por_consulta(env: Entorno) -> list[tuple[Consulta, pd.DataFrame]]:
    """Señales del clasificador por consulta del conjunto A, sin la sentencia fuente en el voto."""
    out = []
    for p in preparar(env):
        c = p.consulta
        if not c.documento_conocido:
            continue
        _, comp, _ = _clasificacion(env, p, excluir={c.doc_id_fuente})
        out.append((c, comp))
    return out


def experimento_ablacion(env: Entorno, log: Callable[[str], None] = print) -> Corrida:
    """Quita un componente del clasificador a la vez y mide la diferencia pareada contra el clasificador
    completo. Si quitar un componente no empeora (Δ ≥ 0 o IC con 0), no está aportando en estos datos."""
    corrida = env.corrida
    log("Calculando componentes del clasificador por consulta...")
    datos = _componentes_por_consulta(env)
    completo = PesosClasificador.desde_config()

    def evaluar(pesos: PesosClasificador) -> dict[str, list[float]]:
        out = {k: [] for k in ("hit3_tipo", "hit3_familia", "top1_familia")}
        for c, comp in datos:
            r = resumen_categorias(c.tipo, c.familia, ranking_categorias(comp, pesos), env.clasificador.tipos_de,
                                   env.familias)
            for k in out:
                out[k].append(r[k])
        return out

    base = evaluar(completo)
    filas = [{"variante": "completo (v10)", "n": len(datos),
              **{f"{k}": float(np.mean(v)) if v else float("nan") for k, v in base.items()}}]
    for comp_nombre in (*COMPONENTES, *PENALIZACIONES):
        r = evaluar(completo.sin(comp_nombre))
        fila = {"variante": f"sin {comp_nombre}", "n": len(datos),
                **{k: float(np.mean(v)) if v else float("nan") for k, v in r.items()}}
        for k in ("hit3_tipo", "hit3_familia"):
            d = m.bootstrap_diferencia_pareada(r[k], base[k], config.EVAL_BOOTSTRAP, config.EVAL_SEED)
            fila[f"Δ {k} [IC95]"] = f"{d['diferencia']:+.3f} [{d['ic_inf']:+.3f}, {d['ic_sup']:+.3f}]"
            fila[f"Δ {k} significativa"] = "sí" if d["significativa"] else "no"
        filas.append(fila)
    corrida.tabla(
        "Ablación del clasificador",
        "Cada fila apaga un componente (peso 0 o penalización desactivada). Δ = variante − completo, pareado por "
        "consulta, sin la sentencia fuente en el voto. Δ negativo significativo = el "
        "componente ayuda; Δ ≥ 0 = no aporta (o estorba) en estos datos.", pd.DataFrame(filas))

    todas = pd.concat([comp for _, comp in datos], ignore_index=True) if datos else pd.DataFrame()
    if len(todas) > 2:
        corr = todas[list(COMPONENTES)].corr().round(3)
        corrida.tabla("Correlación entre señales (todos los candidatos)",
                      "Pearson entre las señales del clasificador. Una correlación alta (p. ej. Jaccard–KEM) indica "
                      "redundancia: las dos miden casi lo mismo.", corr.reset_index(names="señal"))
    return corrida


def experimento_pesos(env_dev: Entorno, env_prueba: Entorno | None, objetivo: str = "hit3_familia",
                      paso: float = 0.05, top: int = 10, log: Callable[[str], None] = print) -> Corrida:
    """Búsqueda en malla de pesos (suman 1, paso 0.05) **solo en desarrollo**; el mejor se evalúa una vez en
    prueba contra los pesos v10. Elegir con prueba y reportar con prueba inflaría el resultado (sobreajuste)."""
    corrida = env_dev.corrida
    corrida.parametros.update(objetivo=objetivo, paso=paso)
    base = PesosClasificador.desde_config()
    W = malla_pesos(paso)
    log(f"Malla: {len(W):,} combinaciones de pesos...")
    casos = [caso_malla(comp, c.tipo, c.familia, env_dev.familias, base)
             for c, comp in _componentes_por_consulta(env_dev)]
    res = evaluar_malla(casos, W)
    secundario = "top1_familia" if objetivo != "top1_familia" else "hit3_tipo"
    orden = np.lexsort((-res[secundario], -res[objetivo]))
    v10 = evaluar_malla(casos, base.vector()[None, :])
    filas = [{"rango": "v10", **dict(zip(COMPONENTES, base.vector())), **{k: float(v[0]) for k, v in v10.items()}}]
    for r, i in enumerate(orden[:top], 1):
        filas.append({"rango": r, **dict(zip(COMPONENTES, W[i])), **{k: float(v[i]) for k, v in res.items()}})
    corrida.tabla(f"Mejores pesos en desarrollo (objetivo: {objetivo})",
                  f"n = {len(casos)} consultas de desarrollo. Con pocas consultas muchas combinaciones empatan; el "
                  "orden entre empates es por la métrica secundaria y luego por el orden de la malla.",
                  pd.DataFrame(filas))
    mejor = PesosClasificador(*W[orden[0]], base.penal_longitud, base.penal_tokens)
    corrida.extra["mejores_pesos"] = dict(zip(COMPONENTES, map(float, W[orden[0]])))

    if env_prueba is not None:
        log("Evaluando una sola vez en prueba (v10 vs. mejores de desarrollo)...")
        datos = _componentes_por_consulta(env_prueba)
        filas_p, por_variante = [], {}
        for nombre, pesos in (("v10", base), ("mejores de desarrollo", mejor)):
            rs = [resumen_categorias(c.tipo, c.familia, ranking_categorias(comp, pesos),
                                     env_prueba.clasificador.tipos_de, env_prueba.familias) for c, comp in datos]
            por_variante[nombre] = rs
            fila = {"pesos": nombre, "n": len(rs)}
            for col, k in (("Hit@3 tipo", "hit3_tipo"), ("Hit@3 familia", "hit3_familia"),
                           ("Top-1 familia", "top1_familia")):
                fila.update(_fila_proporcion(col, [r[k] for r in rs]))
            filas_p.append(fila)
        corrida.tabla("Resultado en prueba (una sola vez)",
                      "Pesos v10 contra los mejores encontrados en desarrollo, evaluados en la partición de prueba.",
                      pd.DataFrame(filas_p))
        if datos:
            difs = [{"métrica": k, "n": len(datos),
                     **_fila_diferencia([r[k] for r in por_variante["mejores de desarrollo"]],
                                        [r[k] for r in por_variante["v10"]])}
                    for k in ("hit3_tipo", "hit3_familia", "top1_familia")]
            corrida.tabla("Diferencia pareada en prueba (mejores − v10)",
                          "Si el IC incluye 0, los pesos manuales de v10 ya están cerca del óptimo para estos datos "
                          "y se conservan.", pd.DataFrame(difs))
        corrida.extra["particion_prueba_usada"] = True
    return corrida


# --------------------------------------------------------------------------------------------------------------
# Conceptos clave (LLM)
# --------------------------------------------------------------------------------------------------------------


def calcular_conceptos(consultas: list[Consulta], llm, path: Path = config.CONCEPTOS_CACHE_PATH,
                       rehacer: bool = False, log: Callable[[str], None] = print) -> dict:
    """Precalcula con el LLM los conceptos clave (paso 1 de la app) de cada consulta y los guarda en JSON.
    La clave incluye un hash del texto: si la consulta se edita, se recalcula."""
    from ..pipeline import limpiar_conceptos

    cache = leer_conceptos(path)
    nuevos = 0
    for c in consultas:
        if not c.consulta.strip():
            continue
        clave = clave_conceptos(c)
        if not rehacer and cache.get(c.id, {}).get("clave") == clave:
            continue
        resp = llm.generate(prompts.prompt_conceptos().format(texto_consulta=c.consulta))
        conceptos, ok = limpiar_conceptos(resp, c.consulta)
        cache[c.id] = {"clave": clave, "conceptos": conceptos, "ok": ok, "respuesta": resp,
                       "modelo": llm.settings.model, "prompt": config.PROMPT_CONCEPTOS_VERSION}
        nuevos += 1
        log(f"   {c.id}: {conceptos[:90]}{'' if ok else '  (falló: se usa la consulta)'}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Conceptos: {nuevos} nuevos, {len(cache)} en caché.")
    return cache


# --------------------------------------------------------------------------------------------------------------
# Controles de calidad (LLM)
# --------------------------------------------------------------------------------------------------------------


def resumen_binario(humano: list[bool], sistema: list[bool | None]) -> dict:
    """Matriz de confusión de un control (positivo = deja pasar) contra el humano. Los veredictos no
    interpretables (None) cuentan como SI, porque así se comporta la app (falla abierta), y se reportan aparte."""
    s = [True if v is None else v for v in sistema]
    vp = sum(h and x for h, x in zip(humano, s))
    fp = sum((not h) and x for h, x in zip(humano, s))
    fn = sum(h and not x for h, x in zip(humano, s))
    vn = sum((not h) and not x for h, x in zip(humano, s))
    r = m.precision_recall_f1_binario(vp, fp, fn, vn)
    r.update(VP=vp, FP=fp, FN=fn, VN=vn, no_interpretables=sum(v is None for v in sistema))
    r["tasa_no_interpretables"] = r["no_interpretables"] / len(sistema) if sistema else 0.0
    return r


def _filas_qc(nombre: str, r: dict, recall_aciertos: int, n_relevantes: int) -> dict:
    lo, hi = m.wilson_ic(recall_aciertos, n_relevantes)
    return {"variante": nombre, "n": r["n"], "VP": r["VP"], "FP": r["FP"], "FN": r["FN"], "VN": r["VN"],
            "precisión": r["precision"], "recall": r["recall"], "recall IC95": f"[{lo:.3f}, {hi:.3f}]",
            "especificidad": r["especificidad"], "F1": r["f1"], "kappa": r["kappa"],
            "no interpretables": r["tasa_no_interpretables"]}


def experimento_qc(env: Entorno, llm, variantes: Iterable[str] = ("inicio", "fragmento_similar"),
                   log: Callable[[str], None] = print) -> Corrida:
    """QC_1 sobre los pares de juicios.csv (relevante = relevancia ≥ 1) y QC_2 sobre las fichas anotadas."""
    from ..embeddings import embed_query
    from ..pipeline import parse_veredicto
    from .conjuntos import cargar_fichas_anotadas

    corrida = env.corrida
    consultas = {c.id: c for c in env.consultas}
    pares = [(cid, d, rel) for cid, docs in env.juicios.items() if cid in consultas for d, rel in docs.items()
             if d in env.store.corpus.index]
    corrida.parametros["variantes"] = list(variantes)
    if not pares:
        corrida.advertir("No hay pares juzgados (juicios.csv) para las consultas de esta partición: QC_1 por medir.")
    else:
        embs: dict[str, np.ndarray] = {}
        filas = []
        humano = [rel >= 1 for _, _, rel in pares]
        for variante in variantes:
            log(f"QC_1 ({variante}) sobre {len(pares)} pares...")
            sistema = []
            for i, (cid, doc_id, _) in enumerate(pares, 1):
                consulta = consultas[cid].consulta
                if variante == "fragmento_similar":
                    q = embs.setdefault(cid, embed_query(consulta))
                    fragmento = env.store.fragmento_mas_similar(doc_id, q)
                else:
                    fragmento = env.store.corpus.loc[doc_id, "texto"][:2500]
                resp = llm.generate(prompts.PROMPT_VERIFICACION_RELEVANCIA.format(
                    texto_consulta=consulta, fragmento_documento=fragmento))
                sistema.append(parse_veredicto(resp))
                if i % 25 == 0:
                    log(f"   {i}/{len(pares)}")
            r = resumen_binario(humano, sistema)
            filas.append(_filas_qc(f"QC_1 · {variante}", r, r["VP"], r["VP"] + r["FN"]))
        corrida.tabla(
            "QC_1 (relevancia) contra el juicio humano",
            "Positivo = QC_1 deja pasar el documento. Recall = fracción de documentos relevantes que conserva (lo que "
            "falta son descartes injustos; es el criterio principal para la defensa). Los veredictos no "
            "interpretables cuentan como SI (así se comporta la app) y su tasa se reporta aparte.",
            pd.DataFrame(filas))

    fichas = [f for f in cargar_fichas_anotadas() if f.get("consulta_id") in consultas and "ficha_aceptable" in f]
    if not fichas:
        corrida.advertir("No hay fichas anotadas con consulta_id y ficha_aceptable: QC_2 por medir.")
        return corrida
    log(f"QC_2 sobre {len(fichas)} fichas anotadas...")
    humano, sistema = [], []
    for f in fichas:
        informe = generar_ficha(env.store.corpus.loc[f["doc_id"], "texto"], llm)["informe"]
        resp = llm.generate(prompts.PROMPT_VERIFICACION_POST_REPORTE.format(
            texto_consulta=consultas[f["consulta_id"]].consulta, informe_generado=informe))
        humano.append(bool(f["ficha_aceptable"]))
        sistema.append(parse_veredicto(resp))
    r = resumen_binario(humano, sistema)
    corrida.tabla("QC_2 (anti-alucinación) contra el juicio humano",
                  "Positivo = QC_2 acepta la ficha; verdad = el anotador la marcó como aceptable.",
                  pd.DataFrame([_filas_qc("QC_2", r, r["VP"], r["VP"] + r["FN"])]))
    return corrida


def experimento_qc_diagnostico(store, clasificador, llm, k: int = 5, log: Callable[[str], None] = print) -> Corrida:
    """Diagnóstico de QC_1 SIN anotaciones: con los 16 ejemplos penales de la app, qué fracción de sus k primeros
    candidatos descarta cada versión del prompt y cada contexto, y en cuántas consultas descarta todo.

    No mide calidad (no sabemos cuáles descartes son correctos): solo muestra qué tan estricto es el control.
    La calidad se mide con `qc` contra juicios humanos."""
    from ..embeddings import embed_query
    from ..pipeline import combinar_candidatos, parse_veredicto
    from .regresion import EJEMPLOS_PENALES, conceptos_para, leer_conceptos

    cache = leer_conceptos()
    pares = []
    for e in EJEMPLOS_PENALES:
        conceptos = conceptos_para(e.texto, "v11", cache)
        q = embed_query(e.texto)
        lexica = f"{e.texto} {conceptos}"
        amplios = store.buscar(e.texto, query_embedding=q, consulta_lexica=lexica)
        tabla = clasificador.clasificar(e.texto, conceptos, q, recuperados=amplios).tabla
        filtro = [t for c in tabla["tipo"].head(config.NUM_CATEGORIAS_FILTRO) for t in clasificador.tipos_de(c)]
        estrictos = store.buscar(e.texto, tipos=filtro, query_embedding=q, consulta_lexica=lexica)
        candidatos, _ = combinar_candidatos(estrictos, amplios, config.RAG_NUM_RESULTS_DESIRED)
        pares += [(e, q, c) for c, _ in candidatos[:k]]
    filas = []
    for version in ("v10", "v11"):
        for contexto in ("inicio", "fragmento_similar"):
            log(f"   QC_1 {version} · {contexto} ({len(pares)} pares)...")
            veredictos: dict[str, list] = {}
            for e, q, c in pares:
                frag = (store.fragmento_mas_similar(c.doc_id, q, config.QC1_MAX_CHARS) if contexto == "fragmento_similar"
                        else c.texto[:config.QC1_MAX_CHARS])
                resp = llm.generate(prompts.prompt_qc1(version).format(texto_consulta=e.texto, fragmento_documento=frag))
                veredictos.setdefault(e.etiqueta, []).append(parse_veredicto(resp))
            todos = [v for vs in veredictos.values() for v in vs]
            filas.append({"prompt QC_1": version, "contexto": contexto, "pares": len(todos),
                          "descartados (NO)": sum(v is False for v in todos) / len(todos),
                          "ambiguos": sum(v is None for v in todos) / len(todos),
                          f"consultas con los {k} descartados": sum(all(v is False for v in vs)
                                                                    for vs in veredictos.values())})
    corrida = Corrida("qc-diagnostico", parametros={"k": k, "consultas": "16 ejemplos penales de la app"})
    corrida.tabla("Tasa de descarte de QC_1 por versión y contexto",
                  f"Para los {k} primeros candidatos de cada uno de los 16 ejemplos penales. Solo diagnóstico: un "
                  "descarte puede ser correcto o injusto; la calidad se mide con `qc` contra juicios humanos.",
                  pd.DataFrame(filas), diagnostico=True)
    return corrida


# --------------------------------------------------------------------------------------------------------------
# Generación de fichas (LLM)
# --------------------------------------------------------------------------------------------------------------


def generar_ficha(texto: str, llm, version_prompt: str | None = None) -> dict:
    """Extracción + informe penal de una sentencia, como en el paso 4 de la app (sin QC)."""
    from ..pipeline import recortar_contexto

    p_ext, p_rep = prompts.prompts_penal(version_prompt)
    contexto = recortar_contexto(texto, llm.settings.max_context_chars)
    extraccion = llm.generate(p_ext.format(contexto=contexto))
    informe = llm.generate(p_rep.format(hechos=extraccion))
    return {"extraccion": extraccion, "informe": informe, "chars_leidos": len(contexto), "chars_total": len(texto)}


def comparar_ficha(extraccion: str, anotada: dict, texto_sentencia: str) -> dict:
    """Compara una ficha generada con su anotación: exactitud por campo, F1 de artículos y precisión de
    artículos contra el texto de la sentencia (automática: un artículo que no aparece en la sentencia es una
    alucinación segura; uno que sí aparece puede seguir siendo el equivocado)."""
    datos = datos_penales(extraccion)
    # Toda la ficha: los artículos también aparecen en "Delito(s)" ("art. 233 en relación con el 236").
    arts_ficha = extraer_articulos(extraccion)
    arts_texto = extraer_articulos(texto_sentencia)
    out = {"doc_id": anotada["doc_id"],
           "sentido_generado": datos["sentido_fallo"], "sentido_anotado": normalizar_sentido(anotada.get("sentido_fallo")),
           "etapa_generada": datos["etapa"], "etapa_anotada": anotada.get("etapa"),
           "delito_generado": datos["delito"], "delito_anotado": anotada.get("delito"),
           "articulos_ficha": sorted(arts_ficha), "articulos_anotados": sorted(extraer_articulos(
               " ".join(f"art. {a}" for a in anotada.get("articulos", []))))}
    out["sentido_ok"] = out["sentido_generado"] == out["sentido_anotado"] if out["sentido_anotado"] else None
    out["etapa_ok"] = coincide_texto(datos["etapa"], anotada.get("etapa")) if anotada.get("etapa") else None
    out["delito_ok"] = coincide_texto(datos["delito"], anotada.get("delito")) if anotada.get("delito") else None
    if anotada.get("articulos") is not None:
        out.update({f"articulos_{k}": v for k, v in f1_conjuntos(arts_ficha, set(out["articulos_anotados"])).items()})
    out["articulos_en_texto"] = (len(arts_ficha & arts_texto) / len(arts_ficha)) if arts_ficha else None
    out["articulos_inexistentes"] = sorted(arts_ficha - arts_texto)
    return out


def afirmaciones(doc_id: str, informe: str) -> list[dict]:
    """Divide una ficha en afirmaciones revisables (viñetas y oraciones) para la revisión de alucinaciones."""
    out = []
    for linea in informe.splitlines():
        t = linea.strip().lstrip("-*• ").strip()
        if len(t) < 25 or t.startswith("#"):
            continue
        for oracion in [o.strip() for o in t.split(". ") if len(o.strip()) >= 25]:
            out.append({"doc_id": doc_id, "afirmacion": oracion, "alucinacion (0/1)": "", "comentario": ""})
    return out


def experimento_generacion(fichas_anotadas: list[dict], corpus: pd.DataFrame, llm, corrida: Corrida,
                           version_prompt: str | None = None, n_afirmaciones: int = 100,
                           log: Callable[[str], None] = print) -> tuple[Corrida, pd.DataFrame, list[dict]]:
    """Genera las fichas del conjunto C, las compara campo por campo y prepara la muestra de afirmaciones."""
    corrida.parametros.update(version_prompt=version_prompt or getattr(config, "PROMPT_PENAL_VERSION", "v10"),
                              n_afirmaciones=n_afirmaciones)
    textos = corpus.set_index("id")["texto"]
    filas, generadas, todas_af = [], [], []
    for i, f in enumerate(fichas_anotadas, 1):
        if f["doc_id"] not in textos.index:
            corrida.advertir(f"{f['doc_id']} no está en el corpus; se omite.")
            continue
        log(f"   [{i}/{len(fichas_anotadas)}] {f['doc_id']}")
        g = generar_ficha(textos[f["doc_id"]], llm, version_prompt)
        generadas.append({"doc_id": f["doc_id"], **g})
        filas.append(comparar_ficha(g["extraccion"], f, textos[f["doc_id"]]))
        todas_af += afirmaciones(f["doc_id"], g["informe"])
    df = pd.DataFrame(filas)
    resumen = []
    for col, nombre in (("sentido_ok", "Sentido del fallo"), ("etapa_ok", "Etapa procesal"), ("delito_ok", "Delito")):
        vals = [float(v) for v in df.get(col, pd.Series(dtype=object)).dropna()] if len(df) else []
        resumen.append({"campo": nombre, **_fila_proporcion("exactitud", vals)})
    for col, nombre in (("articulos_f1", "Artículos citados (F1 vs. anotación)"),
                        ("articulos_en_texto", "Artículos de la ficha presentes en la sentencia")):
        vals = [float(v) for v in df.get(col, pd.Series(dtype=object)).dropna()] if len(df) else []
        resumen.append({"campo": nombre, **_fila_promedio("exactitud", vals)})
    corrida.tabla("Calidad de la ficha por campo",
                  "Exactitud = coincide con la anotación humana (sentido: valor cerrado; etapa y delito: igual o "
                  "contenido tras normalizar). Artículos: F1 entre conjuntos de números. 'Presentes en la sentencia' "
                  "es automática: un artículo ausente del texto es una alucinación segura.", pd.DataFrame(resumen))
    if len(df):
        detalle = df.copy()
        for col in ("articulos_ficha", "articulos_anotados", "articulos_inexistentes"):
            if col in detalle:
                detalle[col] = detalle[col].map(lambda v: ", ".join(v) if isinstance(v, list) else v)
        corrida.tabla("Detalle por sentencia", "Valores generados vs. anotados.", detalle)
    rng = np.random.default_rng(config.EVAL_SEED)
    muestra = [todas_af[i] for i in sorted(rng.choice(len(todas_af), size=min(n_afirmaciones, len(todas_af)),
                                                      replace=False))] if todas_af else []
    corrida.advertir("Tasa de alucinación por medir: marca la columna 'alucinacion (0/1)' del CSV de afirmaciones.")
    return corrida, pd.DataFrame(muestra), generadas


# --------------------------------------------------------------------------------------------------------------
# Latencia (LLM, sin caché)
# --------------------------------------------------------------------------------------------------------------


def experimento_latencia(env: Entorno, llm_cliente, max_consultas: int = 10,
                         log: Callable[[str], None] = print) -> Corrida:
    """Ejecuta la app completa en N consultas y mide p50/p95 por etapa, llamadas al LLM por consulta,
    descartes por control y veredictos ambiguos."""
    from ..pipeline import PASOS, Copiloto

    corrida = env.corrida
    corrida.parametros["max_consultas"] = max_consultas
    tiempos = {k: [] for k, _ in PASOS}
    totales, llamadas, descartes, ambiguos, fallidas = [], [], Counter(), [], 0
    for i, c in enumerate(env.consultas[:max_consultas], 1):
        llm = LLMContador(llm_cliente)
        cop = Copiloto(env.store, env.clasificador, llm)
        inicio: dict[str, float] = {}
        t0 = time.time()
        resultado = None
        for ev in cop.analizar(c.consulta):
            if ev.tipo == "paso_inicio":
                inicio[ev.paso] = time.time()
            elif ev.tipo == "paso_fin":
                tiempos[ev.paso].append(time.time() - inicio[ev.paso])
            elif ev.tipo == "resultado":
                resultado = ev.datos["resultado"]
        totales.append(time.time() - t0)
        llamadas.append(llm.llamadas)
        if resultado is None:
            fallidas += 1
        else:
            descartes.update(d["motivo"] for d in resultado.descartados)
            ambiguos.append(getattr(resultado, "veredictos_ambiguos", 0))
        log(f"   [{i}] {c.id}: {totales[-1]:.0f} s · {llm.llamadas} llamadas")
    filas = []
    for k, nombre in PASOS:
        if tiempos[k]:
            filas.append({"etapa": nombre, "n": len(tiempos[k]), **m.percentiles(tiempos[k])})
    filas.append({"etapa": "Total por consulta", "n": len(totales), **m.percentiles(totales)})
    corrida.tabla("Latencia (segundos)", "p50 = experiencia típica; p95 = la peor razonable.", pd.DataFrame(filas))
    corrida.tabla("Llamadas al LLM y descartes", "Por consulta ejecutada.", pd.DataFrame([{
        "consultas": len(totales), "sin resultado": fallidas,
        **{f"llamadas {k}": v for k, v in m.percentiles(llamadas).items()},
        "llamadas promedio": float(np.mean(llamadas)) if llamadas else 0.0,
        **{f"descartes {k}": v / max(1, len(totales)) for k, v in descartes.items()},
        "veredictos ambiguos por consulta": float(np.mean(ambiguos)) if ambiguos else 0.0}]))
    import torch

    if torch.cuda.is_available():
        corrida.extra["gpu"] = torch.cuda.get_device_name(0)
        corrida.extra["vram_pico_embeddings_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return corrida


def fmt(r: dict) -> str:  # reexport de conveniencia
    return fmt_ic(r)
