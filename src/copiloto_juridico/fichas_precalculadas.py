"""Fichas penales precalculadas (v11.1): la ficha JSON de cada sentencia penal se genera una sola vez.

Generar fichas durante la consulta era casi todo el tiempo de respuesta. `python -m copiloto_juridico.evaluacion
fichas-batch` las genera para todo el corpus penal y las guarda en `data/fichas/fichas-penal.parquet`, con las
verificaciones deterministas ya aplicadas. La app usa la ficha guardada solo si coinciden la sentencia, su texto
(huella) y la versión (prompt + modelo + parámetros); si no, la genera como antes. Así el abogado anota exactamente
la ficha que la app muestra.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config, prompts
from .fichas import FichaPenal, generar_ficha_json, verificaciones

COLUMNAS = ["doc_id", "huella_texto", "version", "modelo", "digest_modelo", "fecha", "json_valido", "intentos",
            "error", "ficha", "sentido_llm", "sentido_reglas", "articulos_no_encontrados", "verificar",
            "chars_leidos", "chars_total", "segundos"]


def huella_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode()).hexdigest()[:16]


def version_ficha(settings: config.LLMSettings) -> str:
    """Todo lo que cambia la ficha: texto del prompt, modelo, parámetros de muestreo y cuánto texto se lee."""
    base = json.dumps([prompts.PROMPT_FICHA_PENAL_JSON, settings.provider, settings.model, settings.temperature,
                       settings.max_tokens, settings.num_ctx, settings.seed, settings.max_context_chars])
    return "v11-" + hashlib.sha256(base.encode()).hexdigest()[:10]


def cargar(path: Path = config.FICHAS_PRECALCULADAS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNAS)


@lru_cache(maxsize=2)
def _indice(path: str, mtime: float) -> dict[str, dict]:
    return {r["doc_id"]: r for r in cargar(Path(path)).to_dict("records")}


def ficha_guardada(doc_id: str, texto: str, settings: config.LLMSettings,
                   path: Path = config.FICHAS_PRECALCULADAS_PATH) -> FichaPenal | None:
    """La ficha precalculada de una sentencia si es válida y vigente (mismo texto, misma versión); si no, None."""
    if not path.exists():
        return None
    r = _indice(str(path), path.stat().st_mtime).get(doc_id)
    if r is None or not r["json_valido"] or r["huella_texto"] != huella_texto(texto) \
            or r["version"] != version_ficha(settings):
        return None
    return FichaPenal.desde_guardada(r["ficha"])


REINTENTOS_SERVIDOR, ESPERA_REINTENTO = 3, 30


def generar_todas(corpus: pd.DataFrame, llm, path: Path = config.FICHAS_PRECALCULADAS_PATH,
                  limite: int | None = None, guardar_cada: int = 10,
                  log: Callable[[str], None] = print) -> pd.DataFrame:
    """Genera la ficha de cada sentencia penal que no la tenga vigente. Reanudable: guarda cada `guardar_cada`
    fichas y, al volver a correr, salta las que ya están (misma huella de texto y versión)."""
    from .llm import LLMError, digest_modelo
    from .pipeline import recortar_contexto

    version, digest = version_ficha(llm.settings), digest_modelo(llm.settings)
    tabla = cargar(path)
    vigentes = {(d, h) for d, h, v in zip(tabla["doc_id"], tabla["huella_texto"], tabla["version"]) if v == version}
    penal = corpus[corpus["materia"] == "penal"]
    pendientes = [(d, t) for d, t in zip(penal["id"], penal["texto"]) if (d, huella_texto(t)) not in vigentes]
    log(f"Fichas: {len(penal)} sentencias penales, {len(penal) - len(pendientes)} vigentes, "
        f"{len(pendientes)} por generar (versión {version}).")
    filas = {r["doc_id"]: r for r in tabla.to_dict("records")}
    t_ini = time.time()
    for i, (doc_id, texto) in enumerate(pendientes[:limite], 1):
        t0 = time.time()
        contexto = recortar_contexto(texto, llm.settings.max_context_chars)
        for intento in range(1, REINTENTOS_SERVIDOR + 1):
            try:
                ficha, info = generar_ficha_json(llm.generate, contexto)
                break
            except LLMError as e:  # el servidor del LLM se cayó o se reinició: no se pierde el lote
                if intento == REINTENTOS_SERVIDOR:
                    _guardar(filas, path)
                    raise
                log(f"   ⚠️ {doc_id}: {e} · reintento {intento} en {ESPERA_REINTENTO} s")
                time.sleep(ESPERA_REINTENTO)
        v = verificaciones(ficha.campos_markdown(), texto) if ficha else {
            "sentido_reglas": None, "sentido_llm": None, "articulos_no_encontrados": [], "verificar": []}
        filas[doc_id] = {
            "doc_id": doc_id, "huella_texto": huella_texto(texto), "version": version, "modelo": llm.settings.model,
            "digest_modelo": digest, "fecha": datetime.now().isoformat(timespec="seconds"),
            "json_valido": ficha is not None, "intentos": info["intentos"], "error": info["error"],
            "ficha": ficha.a_json() if ficha else None, "sentido_llm": v["sentido_llm"],
            "sentido_reglas": v["sentido_reglas"], "articulos_no_encontrados": json.dumps(v["articulos_no_encontrados"]),
            "verificar": json.dumps(v["verificar"], ensure_ascii=False), "chars_leidos": len(contexto),
            "chars_total": len(texto), "segundos": round(time.time() - t0, 1)}
        if i % guardar_cada == 0 or i == len(pendientes[:limite]):
            _guardar(filas, path)
            resta = (time.time() - t_ini) / i * (len(pendientes[:limite]) - i)
            log(f"   {i}/{len(pendientes[:limite])} · {filas[doc_id]['segundos']:.0f} s la última · "
                f"faltan ~{resta / 3600:.1f} h")
    return cargar(path)


def _guardar(filas: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(list(filas.values()), columns=COLUMNAS).sort_values("doc_id").to_parquet(
        tmp, compression="zstd", index=False)
    tmp.replace(path)  # atómico: si se corta a media escritura, queda la versión anterior


def actualizar_verificaciones(corpus: pd.DataFrame, path: Path = config.FICHAS_PRECALCULADAS_PATH) -> int:
    """Recalcula las verificaciones deterministas (sentido por reglas, artículos no encontrados, marcas "verificar")
    de las fichas guardadas cuyo texto no cambió, sin LLM: así un cambio en las reglas llega al diagnóstico sin
    regenerar fichas. (La app ya las calcula al mostrar cada ficha.) Devuelve cuántas filas cambiaron."""
    tabla = cargar(path)
    textos = corpus.set_index("id")["texto"].to_dict()
    filas, cambios = {}, 0
    for r in tabla.to_dict("records"):
        texto = textos.get(r["doc_id"])
        if texto is not None and isinstance(r["ficha"], str) and r["huella_texto"] == huella_texto(texto):
            v = verificaciones(FichaPenal.desde_guardada(r["ficha"]).campos_markdown(), texto)
            nuevo = {"sentido_llm": v["sentido_llm"], "sentido_reglas": v["sentido_reglas"],
                     "articulos_no_encontrados": json.dumps(v["articulos_no_encontrados"]),
                     "verificar": json.dumps(v["verificar"], ensure_ascii=False)}
            cambios += any((r[k] if isinstance(r[k], str) else None) != x for k, x in nuevo.items())
            r.update(nuevo)
        filas[r["doc_id"]] = r
    _guardar(filas, path)
    return cambios
