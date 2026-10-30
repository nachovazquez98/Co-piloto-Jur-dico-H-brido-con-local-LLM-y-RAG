"""Criterio de balance del corpus.

Un *tipo* de sentencia es la pareja (materia, delito/acción) tal como la publica el STJ Jalisco,
p. ej. `penal/robo-calificado` o `civil/accion-reivindicatoria_otras`.

Criterio: cada tipo aporta como máximo `MAX_SENTENCIAS_POR_TIPO` (30) sentencias. La meta de un tipo es
`min(30, sentencias publicadas de ese tipo)`; el corpus está balanceado cuando todos los tipos alcanzan su meta.
Dentro de un tipo con más sentencias de las necesarias se eligen repartidas en el tiempo (por fecha),
para que no sean todas del mismo periodo.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def slugify(value: str) -> str:
    """Mismo slug que usa el descargador para nombrar los PDFs (Fase 0 del notebook)."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def _normalizar_accion(accion: str) -> str:
    # El slug de los nombres de archivo puede dejar guiones junto a los separadores ("servicios-_otros").
    return re.sub(r"-*_-*", "_", accion.strip().lower()).strip("-_")


def tipo_key(materia: str, delito_o_accion: str) -> str:
    return f"{slugify(materia)}/{_normalizar_accion(delito_o_accion)}"


def load_catalogo(path: Path = config.CATALOGO_PATH) -> pd.DataFrame:
    """Catálogo de tipos publicado por el STJ con el número de sentencias disponibles de cada uno."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for nombre, total in csv.reader(f):
            partes = nombre.split("_")
            materia = slugify(partes[0])
            accion = _normalizar_accion("_".join(slugify(p) for p in partes[1:]))
            rows.append({"tipo": f"{materia}/{accion}", "materia": materia, "delito_o_accion": accion,
                         "nombre": nombre, "disponibles": int(total)})
    df = pd.DataFrame(rows).groupby(["tipo", "materia", "delito_o_accion"], as_index=False).agg(
        nombre=("nombre", "first"), disponibles=("disponibles", "sum"))
    df["meta"] = df["disponibles"].clip(upper=config.MAX_SENTENCIAS_POR_TIPO)
    return df


def select_balanced(corpus: pd.DataFrame, max_por_tipo: int = config.MAX_SENTENCIAS_POR_TIPO) -> pd.DataFrame:
    """Aplica el tope por tipo eligiendo sentencias repartidas uniformemente en el tiempo."""
    partes = []
    for _, grupo in corpus.sort_values(["tipo", "fecha", "id"]).groupby("tipo", sort=True):
        if len(grupo) > max_por_tipo:
            idx = np.linspace(0, len(grupo) - 1, num=max_por_tipo).round().astype(int)
            grupo = grupo.iloc[sorted(set(idx))]
        partes.append(grupo)
    return pd.concat(partes, ignore_index=True)


def balance_report(corpus: pd.DataFrame, catalogo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla por tipo: sentencias en el corpus, meta y si el tipo ya está completo."""
    actual = corpus.groupby(["tipo", "materia", "delito_o_accion"]).size().rename("en_corpus").reset_index()
    if catalogo is None:
        actual["meta"] = config.MAX_SENTENCIAS_POR_TIPO
        rep = actual
    else:
        rep = catalogo.merge(actual, on=["tipo", "materia", "delito_o_accion"], how="outer")
        rep["en_corpus"] = rep["en_corpus"].fillna(0).astype(int)
        rep["meta"] = rep["meta"].fillna(rep["en_corpus"].clip(upper=config.MAX_SENTENCIAS_POR_TIPO)).astype(int)
    rep["faltan"] = (rep["meta"] - rep["en_corpus"]).clip(lower=0)
    rep["completo"] = rep["faltan"] == 0
    return rep.sort_values(["materia", "tipo"]).reset_index(drop=True)
