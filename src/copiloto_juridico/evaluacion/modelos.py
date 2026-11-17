"""Comparación de modelos locales de Ollama (v11.1): `python -m copiloto_juridico.evaluacion modelos`.

Solo con los modelos de `config.MODELOS_COMPARACION` que ya están instalados en Ollama: los que faltan se omiten
con un aviso y **no se descargan**. Cada modelo corre con la misma semilla, temperatura 0 y los mismos insumos:

- Latencia p50/p95 y llamadas al LLM: la app completa sobre los primeros N ejemplos penales, generando las fichas
  en el momento (sin fichas precalculadas, que solo existen para el modelo de la app).
- Fichas sin anotación, sobre la misma muestra de sentencias penales (semilla de evaluación): JSON válido, sentido
  del fallo igual al de los resolutivos (reglas), artículos citados que no están en la sentencia y "dato no
  disponible".
- VRAM máxima de la GPU durante el modelo (`nvidia-smi`, toda la GPU; "no disponible" si no hay).
- Si el conjunto C ya está anotado: exactitud por campo y F1 de artículos contra la anotación. Eso usa la hoja
  de evaluación final, así que pasa por el control del congelado (una vez por congelado).

Antes de cada modelo se descargan de la memoria de Ollama los que estén cargados, para medir cada uno solo: córrelo
sin otra tarea usando Ollama (cortaría sus peticiones en curso).
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .. import config
from . import metricas as m
from .reporte import Corrida


def instalados(settings: config.LLMSettings) -> dict[str, str]:
    """{nombre: digest} de los modelos que Ollama ya tiene descargados."""
    r = requests.get(settings.base_url.rstrip("/") + "/api/tags", timeout=5)
    r.raise_for_status()
    out = {}
    for x in r.json().get("models", []):
        for nombre in {x.get("name"), x.get("model")} - {None}:
            out[nombre] = x.get("digest", "")
    return out


def descargar_de_memoria(settings: config.LLMSettings) -> None:
    """Saca de la VRAM los modelos cargados en Ollama (keep_alive 0), para medir cada modelo solo."""
    base = settings.base_url.rstrip("/")
    try:
        for x in requests.get(base + "/api/ps", timeout=5).json().get("models", []):
            requests.post(base + "/api/generate", json={"model": x["name"], "keep_alive": 0}, timeout=60)
    except (requests.RequestException, ValueError, KeyError):
        pass


class MonitorVRAM:
    """Memoria usada de la GPU (MiB), muestreada cada segundo con nvidia-smi mientras dura el bloque `with`."""

    def __init__(self, intervalo: float = 1.0):
        self.intervalo, self.muestras = intervalo, []
        self.disponible = shutil.which("nvidia-smi") is not None
        self._alto = threading.Event()

    def _leer(self) -> float | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return sum(float(x) for x in out.split())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _bucle(self):
        while not self._alto.wait(self.intervalo):
            v = self._leer()
            if v is not None:
                self.muestras.append(v)

    def __enter__(self):
        if self.disponible:
            self._hilo = threading.Thread(target=self._bucle, daemon=True)
            self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._alto.set()
        if self.disponible:
            self._hilo.join()

    @property
    def maximo(self) -> float | None:
        return max(self.muestras) if self.muestras else None


def muestra_sentencias(corpus: pd.DataFrame, n: int, semilla: int = config.EVAL_SEED) -> pd.DataFrame:
    """Las mismas n sentencias penales para todos los modelos."""
    penal = corpus[corpus["materia"] == "penal"].sort_values("id")
    idx = np.random.default_rng(semilla).choice(len(penal), size=min(n, len(penal)), replace=False)
    return penal.iloc[sorted(idx)]


def _pct(k: int, d: int) -> str:
    return f"{k / d:.1%} ({k}/{d})" if d else "—"


def fila_fichas(tabla: pd.DataFrame) -> dict:
    from .experimentos import indicadores_fichas

    i = indicadores_fichas(tabla)
    return {"fichas": i["n"], "JSON válido": _pct(i["validas"], i["n"]),
            "sentido = reglas": _pct(i["coinciden"], i["ambos"]),
            "artículos no encontrados": _pct(i["faltan"], i["citados"]),
            "dato no disponible": _pct(i["vacios"], i["campos"]),
            "s por ficha (p50)": m.percentiles(i["segundos"])["p50"] if i["n"] else None}


def fila_anotadas(tabla: pd.DataFrame, anotadas: list[dict], textos: pd.Series) -> dict:
    """Exactitud por campo y F1 de artículos contra el conjunto C (solo las fichas con JSON válido)."""
    from ..fichas import FichaPenal
    from .experimentos import comparar_ficha

    por_doc = {r["doc_id"]: r for r in tabla.to_dict("records")}
    comp = [comparar_ficha(FichaPenal.desde_guardada(por_doc[a["doc_id"]]["ficha"]).campos_markdown(), a,
                           textos[a["doc_id"]])
            for a in anotadas if a["doc_id"] in por_doc and por_doc[a["doc_id"]]["json_valido"]]
    out = {"anotadas": f"{len(comp)}/{len(anotadas)}"}
    for col, nombre in (("sentido_ok", "exactitud sentido"), ("etapa_ok", "exactitud etapa"),
                        ("delito_ok", "exactitud delito")):
        v = [c[col] for c in comp if c.get(col) is not None]
        out[nombre] = _pct(sum(v), len(v))
    f1 = [c["articulos_f1"] for c in comp if c.get("articulos_f1") is not None]
    out["F1 artículos"] = float(np.mean(f1)) if f1 else None
    return out


def comparar(store, clasificador, modelos: list[str], n_consultas: int, n_fichas: int,
             anotadas: list[dict] | None = None, log: Callable[[str], None] = print) -> Corrida:
    from ..fichas_precalculadas import generar_todas
    from ..llm import get_client
    from .conjuntos import Consulta
    from .experimentos import Entorno, experimento_latencia
    from .regresion import EJEMPLOS_PENALES

    base = config.LLMSettings()
    corrida = Corrida("modelos", parametros={"modelos": modelos, "consultas": f"{n_consultas} ejemplos penales",
                                              "fichas": n_fichas, "semilla_llm": base.seed,
                                              "temperatura": base.temperature, "num_ctx": base.num_ctx})
    hay = instalados(base)
    muestra = muestra_sentencias(store.corpus.reset_index(drop=True), n_fichas)
    if anotadas:
        extra = store.corpus.loc[[a["doc_id"] for a in anotadas if a["doc_id"] in store.corpus.index]]
        insumo = pd.concat([muestra, extra[~extra["id"].isin(muestra["id"])]], ignore_index=True)
    else:
        insumo = muestra
    consultas = [Consulta(f"E{i:02d}", e.texto, None, None, None, "desarrollo")
                 for i, e in enumerate(EJEMPLOS_PENALES, 1)][:n_consultas]
    precalculadas, config.FICHAS_PRECALCULADAS = config.FICHAS_PRECALCULADAS, False
    filas, filas_c = [], []
    try:
        for modelo in modelos:
            if modelo not in hay:
                corrida.advertir(f"{modelo} no está instalado en Ollama; se omite (no se descarga solo).")
                log(f"⚠️ {modelo}: no instalado, se omite.")
                continue
            settings = dataclasses.replace(base, model=modelo)
            cliente = get_client(settings)
            descargar_de_memoria(base)
            log(f"== {modelo} ==")
            with MonitorVRAM() as vram, tempfile.TemporaryDirectory() as tmp:
                tabla = generar_todas(insumo, cliente, path=Path(tmp) / "fichas.parquet", log=log)
                lat = Corrida("latencia")
                env = Entorno(store, clasificador, {}, consultas, {}, {}, lat)
                experimento_latencia(env, cliente, max_consultas=n_consultas, log=log)
            total = lat.tablas["Latencia (segundos)"][1].iloc[-1]
            llamadas = lat.tablas["Llamadas al LLM y descartes"][1].iloc[0]
            de_muestra = tabla[tabla["doc_id"].isin(muestra["id"])]
            filas.append({"modelo": modelo, "digest": hay[modelo][:12],
                          "latencia p50 (s)": total["p50"], "latencia p95 (s)": total["p95"],
                          "llamadas al LLM (promedio)": llamadas["llamadas promedio"],
                          **fila_fichas(de_muestra),
                          "VRAM máxima (MiB)": vram.maximo if vram.disponible else "no disponible"})
            if anotadas:
                filas_c.append({"modelo": modelo, **fila_anotadas(tabla, anotadas, store.corpus["texto"])})
    finally:
        config.FICHAS_PRECALCULADAS = precalculadas
        descargar_de_memoria(base)
    corrida.tabla("Modelos locales",
                  f"Misma semilla ({base.seed}), temperatura {base.temperature}, mismas {n_consultas} consultas de "
                  f"ejemplo y mismas {len(muestra)} sentencias. Latencia de la app completa generando las fichas en el "
                  "momento (sin precalculadas). Fichas sin anotación: un desacuerdo con las reglas no es "
                  "necesariamente un error del modelo. VRAM: toda la GPU (incluye embeddings si están en GPU). "
                  "Diagnóstico: no es la partición de prueba.", pd.DataFrame(filas), diagnostico=True)
    if anotadas:
        corrida.tabla("Modelos contra el conjunto C anotado",
                      "Exactitud por campo y F1 de artículos contra la anotación del abogado (fichas con JSON "
                      "válido). Usa la hoja de evaluación final: no elijas el modelo con esta tabla y después la "
                      "reportes como evaluación.", pd.DataFrame(filas_c))
    return corrida


# --------------------------------------------------------------------------------------------------------------
# v11.2: qué modelo rinde mejor en cada tarea de la app, contra la referencia automática (`modelos-tareas`)
# --------------------------------------------------------------------------------------------------------------

MODELOS_DIR = config.EVAL_DIR / "modelos"
REGLA_CAMBIO = ("Regla fijada antes de correr: en cada tarea, un modelo sustituye al de la app solo si su diferencia "
                "pareada en la métrica principal tiene IC95 > 0 en la partición de desarrollo; si no, se queda el de "
                "la app. La elección se confirma una sola vez en la partición de prueba.")


def _slug(modelo: str) -> str:
    return modelo.replace(":", "-").replace("/", "-")


def familia_modelo(modelo: str) -> str:
    """'qwen2.5:7b' -> 'qwen', 'gemma2:9b' -> 'gemma', 'llama3.1:8b' -> 'llama'."""
    return re.match(r"[a-z]+", modelo.lower()).group()


def cache_conceptos(modelo: str) -> Path:
    """Los conceptos del modelo de la app van en la caché de siempre; los de los demás, en una propia."""
    return config.CONCEPTOS_CACHE_PATH if modelo == config.LLMSettings().model else \
        config.CONCEPTOS_CACHE_PATH.with_name(f"conceptos-{_slug(modelo)}.json")


def puntaje_ficha(fila: dict | None) -> float:
    """Fracción de campos correctos de una ficha contra la referencia (los campos sin referencia no cuentan). Sin
    ficha válida = 0: un JSON inválido deja la ficha vacía en la app."""
    if fila is None:
        return 0.0
    ok = [fila[f"{c}_ok"] for c in ("familia", "acto_impugnado", "apelante", "sentido") if fila[f"{c}_ok"] is not None]
    return float(np.mean(ok)) if ok else 0.0


def decision(fila: dict) -> str:
    if fila["¿significativa?"] != "sí":
        return "sin diferencia: se queda el de la app"
    return "mejor que el de la app" if fila["Δ"] > 0 else "peor que el de la app"


def comparar_tareas(env, modelos: list[str], n_fichas: int = 150, n_pares: int = 600,
                    log: Callable[[str], None] = print) -> Corrida:
    """Cada modelo hace las tareas medibles de la app con los mismos insumos (partición de desarrollo):

    - conceptos: Hit@3 de familia de la clasificación con sus conceptos (leave-one-out; no depende de qué
      documentos se juzgaron, así que no favorece al modelo con que se armaron los pares);
    - QC_1: acierto por par contra la referencia de relevancia (relevante = 1 o 2), en los mismos pares;
    - fichas: fracción de campos correctos contra la referencia de sentencias, en las mismas sentencias. Análisis
      de sensibilidad: la referencia sin el juez de la misma familia del modelo (los modelos de una familia
      tienden a coincidir entre sí).

    El reporte final no tiene métrica automática y no se compara aquí."""
    from ..fichas_precalculadas import generar_todas
    from ..llm import get_client
    from ..pipeline import parse_veredicto
    from .. import prompts
    from . import referencias as R
    from .experimentos import (LLMConCache, Preparada, _clasificacion, _fila_diferencia, calcular_conceptos,
                               conceptos_de, preparar, resumen_binario, resumen_categorias)

    corrida = env.corrida
    base = config.LLMSettings()
    hay = instalados(base)
    modelos = [base.model] + [x for x in modelos if x != base.model]
    for x in [x for x in modelos if x not in hay]:
        corrida.advertir(f"{x} no está instalado en Ollama; se omite (no se descarga solo).")
    modelos = [x for x in modelos if x in hay]
    if not R.REFERENCIAS_PATH.exists():
        raise SystemExit("❌ Faltan las referencias: corre primero `jueces` y `referencias`.")

    # Insumos fijos, iguales para todos los modelos
    rng = np.random.default_rng(config.EVAL_SEED)
    ref = pd.read_csv(R.REFERENCIAS_PATH, dtype=str)
    ref = ref.astype(object).where(ref.notna(), None)
    jueces = R.usables(R.validacion_jueces(ref, config.MODELOS_JUECES))
    corpus = env.store.corpus
    dev = sorted(set(ref.loc[ref["particion"] == "desarrollo", "doc_id"]) & set(corpus["id"]))
    ids = sorted(rng.choice(dev, size=min(n_fichas, len(dev)), replace=False))
    muestra = corpus[corpus["id"].isin(ids)].reset_index(drop=True)
    consultas = {c.id: c for c in env.consultas}
    pares = sorted((cid, d, rel) for cid, docs in env.juicios.items() if cid in consultas for d, rel in docs.items()
                   if d in corpus.index)
    if len(pares) > n_pares:
        pares = [pares[i] for i in sorted(rng.choice(len(pares), size=n_pares, replace=False))]
    relevante = [rel >= 1 for _, _, rel in pares]
    conocidas = [p for p in preparar(env) if p.consulta.documento_conocido]
    q_de = {p.consulta.id: p.q for p in preparar(env)}
    corrida.parametros.update({"modelos": modelos, "particion": "desarrollo", "fichas": len(ids), "pares_qc1": len(pares),
                               "consultas_conceptos": len(conocidas), "jueces_de_la_referencia": jueces,
                               "qc1": [config.QC1_CONTEXTO, config.PROMPT_QC1_VERSION], "regla": REGLA_CAMBIO})
    log(f"Comparación por tarea: {len(modelos)} modelos · {len(conocidas)} consultas · {len(pares)} pares · "
        f"{len(ids)} sentencias (desarrollo).")

    filas, por_modelo = [], {}
    try:
        for modelo in modelos:
            settings = dataclasses.replace(base, model=modelo)
            descargar_de_memoria(base)
            log(f"== {modelo} ==")
            cliente = LLMConCache(get_client(settings))

            # 1. Conceptos -> clasificación
            cache = calcular_conceptos(env.consultas, cliente, path=cache_conceptos(modelo), log=lambda _m: None)
            conceptos, _ = conceptos_de(env.consultas, cache)
            hits = []
            for p in conocidas:
                c = p.consulta
                p2 = Preparada(c, p.q, conceptos[c.id], f"{c.consulta} {conceptos[c.id]}")
                _, _, rk = _clasificacion(env, p2, excluir={c.doc_id_fuente})
                hits.append(resumen_categorias(c.tipo, c.familia, rk, env.clasificador.tipos_de, env.familias)
                            ["hit3_familia"])
            log(f"   conceptos: Hit@3 familia {np.mean(hits):.3f}")

            # 2. QC_1 con el contexto y el prompt de la app
            veredictos = []
            for i, (cid, d, _) in enumerate(pares, 1):
                consulta = consultas[cid].consulta
                if config.QC1_CONTEXTO == "fragmento_similar":
                    frag = env.store.fragmento_mas_similar(d, q_de[cid], config.QC1_MAX_CHARS)
                else:
                    frag = corpus.loc[d, "texto"][:config.QC1_MAX_CHARS]
                veredictos.append(parse_veredicto(cliente.generate(prompts.prompt_qc1().format(
                    texto_consulta=consulta, fragmento_documento=frag))))
                if i % 100 == 0:
                    log(f"   QC_1: {i}/{len(pares)}")
            qc = resumen_binario(relevante, veredictos)
            acierto_qc = [float((True if v is None else v) == h) for v, h in zip(veredictos, relevante)]

            # 3. Fichas (sin caché de respuestas: son largas). También las del modelo de la app se generan aquí, no
            # se toman las precalculadas: así todos los modelos corren con la misma versión de Ollama.
            path = MODELOS_DIR / _slug(modelo) / "fichas.parquet"
            tabla = generar_todas(muestra, get_client(settings), path=path, log=log)
            tabla = tabla[tabla["doc_id"].isin(ids)]
            comp = {r["doc_id"]: r for r in R.comparar_fichas(tabla, ref).to_dict("records")}
            otros = [j for j in jueces if familia_modelo(j) != familia_modelo(modelo)]
            comp_sin = {r["doc_id"]: r for r in R.comparar_fichas(tabla, R.con_consenso(ref, otros)).to_dict("records")}
            puntos = [puntaje_ficha(comp.get(d)) for d in ids]
            puntos_sin = [puntaje_ficha(comp_sin.get(d)) for d in ids]
            exact = R.tabla_exactitud_fichas(pd.DataFrame(comp.values())).set_index("campo")["exactitud"] if comp \
                else pd.Series(dtype=float)

            por_modelo[modelo] = {"conceptos": hits, "QC_1": acierto_qc, "fichas": puntos}
            filas.append({"modelo": modelo, "familia": familia_modelo(modelo), "digest": hay[modelo][:12],
                          "conceptos: Hit@3 familia": float(np.mean(hits)) if hits else None,
                          "QC_1: acierto": float(np.mean(acierto_qc)) if acierto_qc else None,
                          "QC_1: recall": qc.get("recall"), "QC_1: especificidad": qc.get("especificidad"),
                          "QC_1: no interpretables": qc.get("tasa_no_interpretables"),
                          "fichas: campos correctos": float(np.mean(puntos)),
                          "fichas: sin juez de su familia": float(np.mean(puntos_sin)),
                          **{f"fichas: {c}": exact.get(c) for c in ("familia", "acto_impugnado", "apelante", "sentido")},
                          **{k: v for k, v in fila_fichas(tabla).items() if k in ("JSON válido", "artículos no encontrados",
                                                                                  "s por ficha (p50)")}})
    finally:
        descargar_de_memoria(base)

    corrida.tabla("Modelos por tarea (desarrollo)",
                  "Mismos insumos para todos. Conceptos: Hit@3 de familia de la clasificación (leave-one-out). QC_1: "
                  "acierto contra la referencia de relevancia (los veredictos no interpretables cuentan como SI, como "
                  "en la app). Fichas: fracción de campos correctos por ficha (JSON inválido = 0); familia contra la "
                  "categoría del STJ, el resto contra la mayoría de lectores. 'Sin juez de su familia': la misma "
                  "medida con la referencia sin el juez de la familia del modelo.", pd.DataFrame(filas))
    app = base.model
    comp_filas = []
    for modelo in modelos[1:]:
        for tarea, metrica in (("conceptos", "Hit@3 familia"), ("QC_1", "acierto por par"),
                               ("fichas", "campos correctos por ficha")):
            a, b = por_modelo[modelo][tarea], por_modelo[app][tarea]
            if not a:
                continue
            f = {"tarea": tarea, "A": modelo, "B": app, "métrica": metrica, "n": len(a), **_fila_diferencia(a, b)}
            f["decisión"] = decision(f)
            comp_filas.append(f)
    if comp_filas:
        corrida.tabla("Cada modelo contra el de la app", f"Δ = media(A − B) pareada, IC95 bootstrap. {REGLA_CAMBIO} "
                      "El reporte final no tiene métrica automática: sigue con el modelo de la app.",
                      pd.DataFrame(comp_filas))
    return corrida
