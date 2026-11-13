"""Resultados guardados para la demo (v11.1).

Si una consulta ya se analizó con la misma configuración, la app muestra el resultado guardado de inmediato (una
línea "Resultado guardado el <fecha>"). La clave junta la consulta normalizada, el hash de la configuración
efectiva, los parámetros del LLM y las opciones de la barra lateral, el digest del modelo de Ollama, la huella del
índice y de las fichas precalculadas, y la del código: si cualquiera cambia, se vuelve a analizar.

Se guarda en `data/cache/resultados/` (local, fuera de git). Guarda la consulta y el resultado: si las consultas
traen datos personales, apágala (`COPILOTO_CACHE_RESULTADOS=false`) o borra esa carpeta. La evaluación nunca la usa
(`python -m copiloto_juridico.evaluacion` la desactiva al arrancar y llama al pipeline directamente).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from . import config

CACHE_DIR = config.DATA_DIR / "cache" / "resultados"


def normalizar(consulta: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", consulta)).strip().lower()


def _huella_datos() -> str:
    """Índice (manifiesto: corpus, modelo de embeddings, fragmentos) y fichas precalculadas (tamaño y fecha)."""
    from .datos.indice import MANIFEST

    try:
        indice = (config.INDICE_DIR / MANIFEST).read_text(encoding="utf-8")
    except OSError:
        indice = ""
    p = config.FICHAS_PRECALCULADAS_PATH
    fichas = f"{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.exists() and config.FICHAS_PRECALCULADAS else ""
    return indice + "|" + fichas


def clave(consulta: str, settings: config.LLMSettings, opciones: dict | None = None) -> str:
    from .evaluacion.congelado import hash_configuracion, huella_codigo
    from .llm import digest_modelo
    from .pipeline import Opciones

    partes = {"consulta": normalizar(consulta), "configuracion": hash_configuracion(), "codigo": huella_codigo(),
              "llm": [settings.provider, settings.model, settings.temperature, settings.max_tokens, settings.num_ctx,
                      settings.seed],
              "opciones": dataclasses.asdict(Opciones(**(opciones or {}))), "digest": digest_modelo(settings),
              "datos": _huella_datos()}
    return hashlib.sha256(json.dumps(partes, sort_keys=True, default=str).encode()).hexdigest()[:24]


def buscar(consulta: str, settings: config.LLMSettings, opciones: dict | None = None,
           directorio: Path = CACHE_DIR) -> dict | None:
    """{"resultado", "logs", "fecha"} si la consulta ya se analizó con esta configuración; si no (o si la caché
    está apagada o el archivo no se puede leer), None."""
    if not config.CACHE_RESULTADOS:
        return None
    p = directorio / f"{clave(consulta, settings, opciones)}.pkl"
    try:
        with open(p, "rb") as f:
            return pickle.load(f)  # archivos que escribe esta misma app en el equipo
    except (OSError, pickle.UnpicklingError, AttributeError, EOFError, ImportError):
        return None


def guardar(resultado, logs: list[str], settings: config.LLMSettings, opciones: dict | None = None,
            directorio: Path = CACHE_DIR) -> Path | None:
    if not config.CACHE_RESULTADOS:
        return None
    directorio.mkdir(parents=True, exist_ok=True)
    p = directorio / f"{clave(resultado.consulta, settings, opciones)}.pkl"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"resultado": resultado, "logs": logs, "fecha": datetime.now().isoformat(timespec="minutes")}, f)
    tmp.replace(p)
    return p
