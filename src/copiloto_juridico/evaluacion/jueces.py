"""Jueces LLM de la evaluación automática: sustituyen al anotador humano, que no está disponible.

Tres tareas, con los criterios de la guía de anotación:
- `sentencia`: familia del delito, acto impugnado, apelante y sentido del fallo, leídos del inicio y de los
  resolutivos. La familia se compara con la categoría del STJ (referencia externa) para validar al juez.
- `relevancia`: un par (consulta, sentencia) → mismo delito, mismo problema procesal y grado 0/1/2.
- `consulta`: ¿es una consulta coherente de un defensor? ¿se identifica el delito?

Los jueces son modelos DISTINTOS del de la app (`config.MODELOS_JUECES`) y corren en tandas separadas, un modelo a la
vez (la GPU tiene 8 GB): `python -m copiloto_juridico.evaluacion jueces --modelo qwen2.5:7b`. Cada respuesta se
guarda en `evaluacion/jueces/<modelo>.jsonl` con la clave del prompt (si cambia el texto o el prompt, se vuelve a
juzgar) y la tanda se puede reanudar. La evaluación lee esos archivos: nunca llama al juez.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config, prompts
from ..metadatos import ACTOS, APELANTES, NO_DETERMINADO, _acto, _apelante

TAREAS = ("sentencia", "relevancia", "consulta")
SENTIDOS = ("CONFIRMA", "MODIFICA", "REVOCA", "REPONE")
PARES_PATH = config.JUECES_DIR / "pares.csv"
CHARS_INICIO, CHARS_FINAL = 2500, 2500
REINTENTOS_SERVIDOR, ESPERA_REINTENTO = 3, 30
_RE_RESOLUTIVOS = re.compile(r"r\s?e\s?s\s?u\s?e\s?l\s?v\s?e|p\s?r\s?o\s?p\s?o\s?s\s?i\s?c\s?i\s?o\s?n\s?e\s?s|resolutivos")


@dataclass
class Item:
    tarea: str
    id: str
    prompt: str

    @property
    def clave(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()[:16]


def _norm(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def extracto(texto: str, n_inicio: int = CHARS_INICIO, n_final: int = CHARS_FINAL) -> tuple[str, str]:
    """Inicio del texto y la sección de resolutivos (la última marca "resuelve"/"proposiciones" de la segunda mitad;
    si no hay, el final del texto)."""
    marcas = list(_RE_RESOLUTIVOS.finditer(texto, len(texto) // 2))
    final = texto[marcas[-1].start():marcas[-1].start() + n_final] if marcas else texto[-n_final:]
    return re.sub(r"\s+", " ", texto[:n_inicio]).strip(), re.sub(r"\s+", " ", final).strip()


# --------------------------------------------------------------------------------------------------------------
# Elementos a juzgar
# --------------------------------------------------------------------------------------------------------------


def etiquetas_familias(familias: dict[str, str]) -> list[str]:
    from .familias import etiqueta_familia

    return sorted({etiqueta_familia(f) for f in familias.values()})


def items_sentencias(penal: pd.DataFrame, familias: dict[str, str]) -> list[Item]:
    lista = ", ".join(etiquetas_familias({t: familias[t] for t in penal["tipo"].unique()}))  # solo las penales
    items = []
    for d, t in zip(penal["id"], penal["texto"]):
        inicio, final = extracto(t)
        items.append(Item("sentencia", d, prompts.PROMPT_JUEZ_SENTENCIA.format(familias=lista, inicio=inicio,
                                                                              final=final)))
    return items


def items_relevancia(pares: pd.DataFrame, textos: dict[str, str], tipos: dict[str, str]) -> list[Item]:
    """Como la hoja del abogado, el juez ve la categoría del portal de la sentencia además del texto."""
    from ..clasificador import etiqueta_tipo

    items = []
    for r in pares.itertuples():
        inicio, final = extracto(textos[r.doc_id], 2000, 1500)
        items.append(Item("relevancia", f"{r.consulta_id}|{r.doc_id}", prompts.PROMPT_JUEZ_RELEVANCIA.format(
            consulta=r.consulta, categoria=etiqueta_tipo(tipos[r.doc_id]), sentencia=f"{inicio}\n[…]\n{final}")))
    return items


def items_consultas(consultas) -> list[Item]:
    return [Item("consulta", c.id, prompts.PROMPT_JUEZ_CONSULTA.format(consulta=c.consulta))
            for c in consultas if c.consulta.strip()]


def armar_pares(store, clasificador, consultas, conceptos: dict[str, str], k_app: int = 10,
                k_lineas_base: int = 5) -> pd.DataFrame:
    """Pares (consulta, sentencia) a juzgar, por agrupamiento (pooling): los `k_app` primeros candidatos de la app
    más los `k_lineas_base` primeros de BM25 solo y de embeddings solos, para que las variantes de `recuperacion`
    se comparen sobre sentencias juzgadas y no solo sobre las que eligió la app."""
    from ..embeddings import embed_texts
    from ..pipeline import buscar_candidatos

    elegibles = [c for c in consultas if c.consulta.strip() and not c.sin_respuesta]
    embs = embed_texts([c.consulta for c in elegibles], kind="query")
    filas = []
    for c, q in zip(elegibles, embs):
        lexica = f"{c.consulta} {conceptos.get(c.id, c.consulta)}"
        b = buscar_candidatos(store, clasificador, c.consulta, conceptos.get(c.id, c.consulta), q,
                              config.RAG_NUM_RESULTS_DESIRED)
        ids = [x.doc_id for x, _ in b.candidatos[:k_app]]
        for modo in ("lexica", "densa"):
            ids += [x.doc_id for x in store.buscar(c.consulta, query_embedding=q, modo=modo, consulta_lexica=lexica,
                                                   num_docs=k_lineas_base)]
        for d in dict.fromkeys(ids):
            filas.append({"consulta_id": c.id, "consulta": c.consulta, "doc_id": d})
    return pd.DataFrame(filas, columns=["consulta_id", "consulta", "doc_id"])


# --------------------------------------------------------------------------------------------------------------
# Respuestas: normalización a valores cerrados
# --------------------------------------------------------------------------------------------------------------


def _si_no(v) -> bool | None:
    t = _norm(v)
    return True if t in ("si", "s", "yes", "true") else False if t in ("no", "n", "false") else None


def _cerrado(v, opciones) -> str:
    t = _norm(v)
    for o in opciones:
        if t == _norm(o):
            return o
    return NO_DETERMINADO


def _familia(v, familias: list[str]) -> str:
    """Etiqueta de familia de la respuesta del juez: exacta; si no, la que dan las reglas de la agrupación al nombre
    del delito ("homicidio culposo" -> delitos culposos); si no, la única etiqueta contenida en la respuesta
    ("delitos sexuales (abuso sexual infantil)")."""
    from ..fichas import familia_del_delito
    from .familias import FAMILIA_OTROS, etiqueta_familia

    exacta = _cerrado(v, familias)
    if exacta != NO_DETERMINADO or not familias:
        return exacta
    t = _norm(v)
    f = familia_del_delito(t)
    if f != FAMILIA_OTROS and etiqueta_familia(f) in familias:
        return etiqueta_familia(f)
    contenidas = [x for x in familias if _norm(x) and _norm(x) in t]
    return contenidas[0] if len(contenidas) == 1 else NO_DETERMINADO


def _con_reglas(v, opciones, regla) -> str:
    """Valor cerrado exacto o, si el juez escribió una frase ("auto de vinculación a proceso", "el Ministerio
    Público"), el que le dan las mismas reglas con que se normaliza la ficha de la app."""
    exacto = _cerrado(v, opciones)
    if exacto != NO_DETERMINADO or _norm(v) in ("", "no determinado"):
        return exacto
    r = regla(str(v))
    return r if r in opciones else NO_DETERMINADO


def normalizar(tarea: str, respuesta: str, familias: list[str] | None = None) -> dict | None:
    """Dict con valores cerrados, o None si la respuesta no es JSON o no trae los campos de la tarea."""
    texto = respuesta or ""
    ini, fin = texto.find("{"), texto.rfind("}")
    try:
        d = json.loads(texto[ini:fin + 1]) if ini >= 0 and fin > ini else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict):
        return None
    if tarea == "sentencia":
        if not {"familia", "acto_impugnado", "apelante", "sentido"} <= set(d):
            return None
        sentido = _norm(d["sentido"]).upper()
        hallados = [s for s in SENTIDOS if s[:5] in sentido]  # "SE REVOCA Y SE REPONE" cuenta como REPONE (reglas)
        return {"familia": _familia(d["familia"], familias or []),
                "acto_impugnado": _con_reglas(d["acto_impugnado"], ACTOS, _acto),
                "apelante": _con_reglas(d["apelante"], APELANTES, _apelante),
                "sentido": "REPONE" if "REPONE" in hallados else hallados[0] if hallados else NO_DETERMINADO}
    if tarea == "relevancia":
        try:
            rel = int(d.get("relevancia"))
        except (TypeError, ValueError):
            return None
        if rel not in (0, 1, 2):
            return None
        return {"relevancia": rel, "mismo_delito": _si_no(d.get("mismo_delito")),
                "mismo_problema_procesal": _si_no(d.get("mismo_problema_procesal"))}
    if tarea == "consulta":
        r = {"es_consulta": _si_no(d.get("es_consulta")), "delito_identificable": _si_no(d.get("delito_identificable"))}
        return r if r["es_consulta"] is not None else None
    raise ValueError(f"tarea desconocida: {tarea}")


# --------------------------------------------------------------------------------------------------------------
# Tandas (un modelo a la vez) y lectura
# --------------------------------------------------------------------------------------------------------------


def en_muestra(clave: str, fraccion: float, semilla: int = config.EVAL_SEED) -> bool:
    """Muestreo estable por hash: el mismo id siempre queda dentro o fuera, sin importar qué más haya."""
    return fraccion >= 1 or int(hashlib.sha256(f"{semilla}|{clave}".encode()).hexdigest()[:12], 16) < fraccion * 16 ** 12


def de_la_muestra(it: "Item") -> bool:
    """Sentencias por su id; pares y consultas por la consulta (una consulta entra con todos sus pares)."""
    if it.tarea == "sentencia":
        return en_muestra(it.id, config.MUESTRA_SENTENCIAS)
    return en_muestra(it.id.split("|")[0], config.MUESTRA_CONSULTAS)


def archivo(modelo: str, directorio: Path = config.JUECES_DIR) -> Path:
    return directorio / f"{re.sub(r'[^A-Za-z0-9.]+', '-', modelo)}.jsonl"


def leer(modelo: str, directorio: Path = config.JUECES_DIR) -> dict[tuple[str, str], dict]:
    """{(tarea, id): registro} con la última respuesta guardada de ese juez."""
    p = archivo(modelo, directorio)
    if not p.exists():
        return {}
    out = {}
    for linea in p.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            r = json.loads(linea)
            out[(r["tarea"], r["id"])] = r
    return out


def vigentes(items: list[Item], registros: dict[tuple[str, str], dict]) -> dict[tuple[str, str], dict | None]:
    """Respuesta normalizada de cada elemento si el juez lo juzgó con el mismo prompt (None si no fue válida)."""
    out = {}
    for it in items:
        r = registros.get((it.tarea, it.id))
        if r is not None and r["clave"] == it.clave:
            out[(it.tarea, it.id)] = r["respuesta"]
    return out


def _sin_relevancia(crudo: str) -> dict | None:
    """El JSON del juez de relevancia si es válido salvo por la calificación; None en otro caso."""
    texto = crudo or ""
    ini, fin = texto.find("{"), texto.rfind("}")
    try:
        d = json.loads(texto[ini:fin + 1]) if ini >= 0 and fin > ini else None
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or normalizar("relevancia", texto) is not None:
        return None
    return d


def _completar_relevancia(it: "Item", crudo: str, llm) -> dict | None:
    """Tercer intento: pide solo la clave "relevancia" y la une a la respuesta anterior del mismo juez."""
    from ..llm import LLMError

    try:
        faltante = llm.generate(prompts.PROMPT_JUEZ_RELEVANCIA_FALTANTE.format(prompt=it.prompt, previa=crudo[:500]),
                                json_mode=True)
    except LLMError:  # queda inválido y la siguiente tanda lo vuelve a intentar
        return None
    return normalizar("relevancia", json.dumps({**(_sin_relevancia(crudo) or {}), **_json_o_vacio(faltante)}))


def _json_o_vacio(texto: str) -> dict:
    ini, fin = (texto or "").find("{"), (texto or "").rfind("}")
    try:
        d = json.loads(texto[ini:fin + 1]) if ini >= 0 and fin > ini else {}
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def juzgar(items: list[Item], llm, familias: list[str] | None = None, directorio: Path = config.JUECES_DIR,
           limite: int | None = None, log: Callable[[str], None] = print) -> dict:
    """Juzga los elementos que faltan (o cuyo prompt cambió) y agrega cada respuesta al archivo del juez en cuanto
    la tiene. Si la respuesta no es JSON válido, un reintento; si sigue mal, se guarda como inválida (None)."""
    from ..llm import LLMError, digest_modelo

    modelo = llm.settings.model
    directorio.mkdir(parents=True, exist_ok=True)
    registros = leer(modelo, directorio)
    hechos = vigentes(items, registros)
    # Pares ya juzgados sin la calificación: solo se pide la clave que falta (no se repiten los dos intentos).
    previas = {(it.tarea, it.id): registros[(it.tarea, it.id)]["crudo"] for it in items
               if it.tarea == "relevancia" and (it.tarea, it.id) in hechos and hechos[(it.tarea, it.id)] is None
               and _sin_relevancia(registros[(it.tarea, it.id)]["crudo"]) is not None}
    for k in previas:
        del hechos[k]
    pendientes = [it for it in items if (it.tarea, it.id) not in hechos][:limite]
    digest = digest_modelo(llm.settings)
    log(f"Juez {modelo}: {len(hechos)} ya juzgados, {len(pendientes)} por juzgar.")
    invalidas = 0
    with open(archivo(modelo, directorio), "a", encoding="utf-8") as f:
        for i, it in enumerate(pendientes, 1):
            t0 = time.time()
            respuesta, crudo = None, previas.get((it.tarea, it.id), "")
            for _ in range(0 if crudo else 2):
                for intento in range(1, REINTENTOS_SERVIDOR + 1):
                    try:
                        crudo = llm.generate(it.prompt, json_mode=True)
                        break
                    except LLMError as e:  # el servidor se reinició: se espera y se reintenta, sin perder lo hecho
                        if intento == REINTENTOS_SERVIDOR:
                            raise
                        log(f"   el LLM falló ({e}); reintento en {ESPERA_REINTENTO} s")
                        time.sleep(ESPERA_REINTENTO)
                respuesta = normalizar(it.tarea, crudo, familias)
                if respuesta is not None:
                    break
            if respuesta is None and it.tarea == "relevancia" and _sin_relevancia(crudo) is not None:
                respuesta = _completar_relevancia(it, crudo, llm)
            invalidas += respuesta is None
            f.write(json.dumps({"tarea": it.tarea, "id": it.id, "clave": it.clave, "modelo": modelo, "digest": digest,
                                "respuesta": respuesta, "crudo": "" if respuesta else crudo[:500],
                                "segundos": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(pendientes):
                log(f"   {i}/{len(pendientes)} ({invalidas} respuestas inválidas)")
    return {"modelo": modelo, "digest": digest, "ya_juzgados": len(hechos), "juzgados": len(pendientes),
            "invalidas": invalidas}


def correr(modelo: str, tareas: tuple[str, ...] = TAREAS, limite: int | None = None,
           log: Callable[[str], None] = print, completa: bool = False) -> list[dict]:
    """Una tanda de un juez: saca de la GPU cualquier otro modelo, juzga las tareas pedidas y lo saca al terminar
    (nunca hay dos LLM cargados a la vez). No descarga modelos: si falta, lo dice."""
    import dataclasses

    from ..datos.corpus import load_corpus
    from ..llm import get_client
    from .conjuntos import cargar_todas
    from .familias import cargar_familias
    from .modelos import descargar_de_memoria, instalados

    settings = dataclasses.replace(config.LLMSettings(), model=modelo)
    if modelo not in instalados(settings):
        raise SystemExit(f"❌ {modelo} no está instalado en Ollama. Descárgalo con: ollama pull {modelo}")
    corpus = load_corpus()
    penal = corpus[corpus["materia"] == "penal"]
    familias = cargar_familias(corpus["tipo"].unique().tolist())
    lista_familias = etiquetas_familias({t: familias[t] for t in penal["tipo"].unique()})
    items: list[Item] = []
    if "sentencia" in tareas:
        items += items_sentencias(penal, familias)
    if "relevancia" in tareas:
        if not PARES_PATH.exists():
            raise SystemExit("❌ Faltan los pares: corre primero `evaluacion jueces --pares`.")
        por_id = corpus.set_index("id")
        items += items_relevancia(pd.read_csv(PARES_PATH, dtype=str), por_id["texto"].to_dict(), por_id["tipo"].to_dict())
    if "consulta" in tareas:
        items += items_consultas(cargar_todas(corpus=corpus, familias=familias, permitir_vacias=True))
    if not completa:
        items = [it for it in items if de_la_muestra(it)]
    descargar_de_memoria(settings)
    try:
        llm = get_client(settings)
        return [juzgar([it for it in items if it.tarea == t], llm, lista_familias, limite=limite, log=log)
                for t in tareas]
    finally:
        descargar_de_memoria(settings)


def guardar_pares(log: Callable[[str], None] = print) -> Path:
    """Arma `pares.csv` con todas las consultas (ambas particiones). Sin LLM: usa los conceptos en caché."""
    from ..clasificador import construir_clasificador
    from ..vectorstore import SentenciasStore
    from .conjuntos import cargar_todas
    from .experimentos import conceptos_de, leer_conceptos
    from .familias import cargar_familias
    from .modelos import descargar_de_memoria

    descargar_de_memoria(config.LLMSettings())  # el modelo de embeddings no cabe junto a un LLM en 8 GB
    store = SentenciasStore(log=lambda _m: None)
    corpus = store.corpus
    clas = construir_clasificador(corpus["tipo"].unique().tolist())
    consultas = cargar_todas(corpus=corpus, familias=cargar_familias(corpus["tipo"].unique().tolist()),
                             permitir_vacias=True)
    conceptos, faltan = conceptos_de(consultas, leer_conceptos())
    if faltan:
        log(f"⚠️  {len(faltan)} consultas sin conceptos en caché (se usa la consulta completa); corre `conceptos`.")
    pares = armar_pares(store, clas, consultas, conceptos)
    config.JUECES_DIR.mkdir(parents=True, exist_ok=True)
    pares.to_csv(PARES_PATH, index=False)
    log(f"{len(pares)} pares de {pares['consulta_id'].nunique()} consultas en {PARES_PATH}")
    return PARES_PATH
