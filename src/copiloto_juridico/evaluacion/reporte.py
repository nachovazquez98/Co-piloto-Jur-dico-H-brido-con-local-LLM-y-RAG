"""Guardado reproducible de cada corrida de evaluación.

Cada corrida escribe `evaluacion/resultados/<fecha>_<commit>_<comando>.json` y `.md` con todo lo necesario
para repetirla o auditarla: configuración completa, semilla, modelos, huella del corpus, commit de git (y si
había cambios sin confirmar) y las tablas de resultados. Si alguna consulta estaba en borrador, la corrida
queda marcada como **no reportable**.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=config.ROOT_DIR, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def info_commit() -> dict:
    """Hash corto del commit y si el árbol de trabajo tiene cambios sin confirmar (`sucio`)."""
    commit = _git("rev-parse", "--short", "HEAD") or "sin-git"
    sucio = bool(_git("status", "--porcelain", "--untracked-files=no", "--", "src", "app"))
    return {"commit": commit, "sucio": sucio}


def snapshot_config() -> dict:
    """Todas las constantes en MAYÚSCULAS de `config.py` (sin rutas absolutas ni secretos)."""
    out = {}
    for k in dir(config):
        if not k.isupper():
            continue
        v = getattr(config, k)
        if isinstance(v, Path):
            v = str(v.relative_to(config.ROOT_DIR)) if v.is_relative_to(config.ROOT_DIR) else str(v)
        elif isinstance(v, (set, frozenset)):
            v = sorted(v)
        elif not isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))):
            continue
        out[k] = v
    s = config.LLMSettings()
    out["LLM"] = {"provider": s.provider, "model": s.model, "temperature": s.temperature,
                  "max_tokens": s.max_tokens, "num_ctx": s.num_ctx, "seed": getattr(s, "seed", None)}
    return out


def _limpio(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


@dataclass
class Corrida:
    comando: str
    parametros: dict = field(default_factory=dict)
    tablas: dict[str, tuple[str, pd.DataFrame]] = field(default_factory=dict)
    diagnosticos: set[str] = field(default_factory=set)
    advertencias: list[str] = field(default_factory=list)
    borrador: bool = False
    extra: dict = field(default_factory=dict)
    # Commit y estado del árbol al EMPEZAR la corrida: el código que corre es el de ese momento.
    git: dict = field(default_factory=info_commit)

    def tabla(self, nombre: str, descripcion: str, df: pd.DataFrame, diagnostico: bool = False) -> None:
        """Añade una tabla. `diagnostico=True`: solo va al .md (la interfaz no la muestra)."""
        self.tablas[nombre] = (descripcion, df)
        if diagnostico:
            self.diagnosticos.add(nombre)

    def advertir(self, msg: str) -> None:
        self.advertencias.append(msg)

    def metadatos(self) -> dict:
        from ..datos.indice import MANIFEST

        try:
            manifest = json.loads((config.INDICE_DIR / MANIFEST).read_text())
        except (OSError, ValueError):
            manifest = {}
        return {"comando": self.comando, "fecha": datetime.now().isoformat(timespec="seconds"), **self.git,
                "semilla": config.EVAL_SEED, "parametros": self.parametros,
                "corpus_fingerprint": manifest.get("corpus_fingerprint"),
                "modelo_embeddings": manifest.get("embedding_model"),
                "modelo_llm": config.LLMSettings().model, "borrador": self.borrador,
                "advertencias": self.advertencias, "configuracion": snapshot_config(), **self.extra}

    def guardar(self, directorio: Path = config.RESULTADOS_DIR) -> tuple[Path, Path]:
        meta = self.metadatos()
        directorio.mkdir(parents=True, exist_ok=True)
        base = f"{datetime.now():%Y%m%d-%H%M%S}_{meta['commit']}{'-sucio' if meta['sucio'] else ''}_{self.comando}"
        datos = {**meta, "tablas": {
            n: {"descripcion": d, "columnas": list(df.columns), "diagnostico": n in self.diagnosticos,
                "filas": [{k: _limpio(v) for k, v in fila.items()} for fila in df.to_dict("records")]}
            for n, (d, df) in self.tablas.items()}}
        pj = directorio / f"{base}.json"
        pj.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pm = directorio / f"{base}.md"
        pm.write_text(self.markdown(meta), encoding="utf-8")
        return pj, pm

    def markdown(self, meta: dict | None = None) -> str:
        meta = meta or self.metadatos()
        lineas = [f"# Evaluación: `{self.comando}`", ""]
        if self.borrador:
            lineas += ["> **BORRADOR — no reportable.** Hay consultas marcadas `borrador: true`.", ""]
        lineas += [f"- Fecha: {meta['fecha']}", f"- Commit: `{meta['commit']}`{' (con cambios sin confirmar)' if meta['sucio'] else ''}",
                   f"- Huella del corpus: `{meta['corpus_fingerprint']}` · embeddings `{meta['modelo_embeddings']}`",
                   f"- LLM: `{meta['modelo_llm']}` · semilla {meta['semilla']}",
                   f"- Parámetros: `{json.dumps(self.parametros, ensure_ascii=False)}`", ""]
        for a in self.advertencias:
            lineas.append(f"> ⚠️ {a}")
        if self.advertencias:
            lineas.append("")
        for nombre, (desc, df) in self.tablas.items():
            lineas += [f"## {nombre}", "", desc, "", tabla_markdown(df), ""]
        return "\n".join(lineas)


def tabla_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(sin filas)_"

    def celda(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join(celda(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def ultimo_resultado(comando: str | None = None, directorio: Path = config.RESULTADOS_DIR) -> dict | None:
    """El JSON más reciente (de un comando, o de cualquiera)."""
    patron = f"*_{comando}.json" if comando else "*.json"
    archivos = sorted(directorio.glob(patron)) if directorio.exists() else []
    if not archivos:
        return None
    datos = json.loads(archivos[-1].read_text(encoding="utf-8"))
    datos["_archivo"] = archivos[-1].name
    return datos
