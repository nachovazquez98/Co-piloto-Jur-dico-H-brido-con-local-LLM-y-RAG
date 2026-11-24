"""Comparación de modelos locales de Ollama: `python -m copiloto_juridico.evaluacion modelos`.

Solo con los modelos de `config.MODELOS_COMPARACION` que ya están instalados en Ollama: los que faltan se omiten
con un aviso y **no se descargan**. Cada modelo corre con la misma semilla, temperatura 0 y los mismos insumos:

- Latencia p50/p95 y llamadas al LLM: la app completa sobre los primeros N ejemplos penales, generando las fichas
  en el momento (sin fichas precalculadas, que solo existen para el modelo de la app).
- Fichas sin anotación, sobre la misma muestra de sentencias penales (semilla de evaluación): JSON válido, sentido
  del fallo igual al de los resolutivos (reglas), artículos citados que no están en la sentencia y "dato no
  disponible".
- VRAM máxima de la GPU durante el modelo (`nvidia-smi`, toda la GPU; "no disponible" si no hay).

Antes de cada modelo se descargan de la memoria de Ollama los que estén cargados, para medir cada uno solo: córrelo
sin otra tarea usando Ollama (cortaría sus peticiones en curso).
"""

from __future__ import annotations

import dataclasses
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


def comparar(store, clasificador, modelos: list[str], n_consultas: int, n_fichas: int,
             log: Callable[[str], None] = print) -> Corrida:
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
    consultas = [Consulta(f"E{i:02d}", e.texto, None, None, None, "desarrollo")
                 for i, e in enumerate(EJEMPLOS_PENALES, 1)][:n_consultas]
    precalculadas, config.FICHAS_PRECALCULADAS = config.FICHAS_PRECALCULADAS, False
    filas = []
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
                tabla = generar_todas(muestra, cliente, path=Path(tmp) / "fichas.parquet", log=log)
                lat = Corrida("latencia")
                env = Entorno(store, clasificador, {}, consultas, {}, {}, lat)
                experimento_latencia(env, cliente, max_consultas=n_consultas, log=log)
            total = lat.tablas["Latencia (segundos)"][1].iloc[-1]
            llamadas = lat.tablas["Llamadas al LLM y descartes"][1].iloc[0]
            filas.append({"modelo": modelo, "digest": hay[modelo][:12],
                          "latencia p50 (s)": total["p50"], "latencia p95 (s)": total["p95"],
                          "llamadas al LLM (promedio)": llamadas["llamadas promedio"],
                          **fila_fichas(tabla),
                          "VRAM máxima (MiB)": vram.maximo if vram.disponible else "no disponible"})
    finally:
        config.FICHAS_PRECALCULADAS = precalculadas
        descargar_de_memoria(base)
    corrida.tabla("Modelos locales",
                  f"Misma semilla ({base.seed}), temperatura {base.temperature}, mismas {n_consultas} consultas de "
                  f"ejemplo y mismas {len(muestra)} sentencias. Latencia de la app completa generando las fichas en el "
                  "momento (sin precalculadas). Fichas sin anotación: un desacuerdo con las reglas no es "
                  "necesariamente un error del modelo. VRAM: toda la GPU (incluye embeddings si están en GPU). "
                  "Diagnóstico: no es la partición de prueba.", pd.DataFrame(filas), diagnostico=True)
    return corrida
