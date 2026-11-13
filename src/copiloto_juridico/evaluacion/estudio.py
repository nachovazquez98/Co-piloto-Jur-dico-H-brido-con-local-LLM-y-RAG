"""Estudio de uso (v11.1): ¿la app ahorra tiempo a un abogado frente a la búsqueda manual?

Archivos en `evaluacion/estudio_uso/` (instrucciones en su README.md):
- `casos.jsonl`: 5 casos penales tomados de los ejemplos de la app, `borrador: true` hasta que un abogado los revise;
- `registro.csv`: una fila por participante y caso, con el método asignado (`manual` / `app`) en el orden en que se
  hacen; el participante llena minutos, precedentes útiles encontrados, utilidad 1–5 y comentario.

`python -m copiloto_juridico.evaluacion estudio` resume: mediana de minutos por método, diferencia pareada por caso
(mediana manual − mediana app) con IC bootstrap, precedentes útiles por minuto y utilidad media. Con menos de
`MIN_PARTICIPANTES` participantes el reporte se titula "estudio exploratorio" (cada caso tendría menos de 5
observaciones por método: no alcanza para generalizar).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import metricas as m
from .reporte import Corrida

ESTUDIO_DIR = config.EVAL_DIR / "estudio_uso"
CASOS_PATH = ESTUDIO_DIR / "casos.jsonl"
REGISTRO_PATH = ESTUDIO_DIR / "registro.csv"
COLUMNAS = ["participante", "caso", "metodo", "minutos", "precedentes_utiles", "utilidad", "comentario"]
METODOS = ("manual", "app")
MIN_PARTICIPANTES = 10
# Ejemplos penales de la app usados como casos (delito y acto distintos, todos con sentencias en el corpus).
EJEMPLOS_ESTUDIO = ("Robo calificado", "Homicidio calificado", "Violencia familiar", "Fraude", "Lesiones calificadas")


def casos_desde_ejemplos() -> list[dict]:
    from ..ejemplos import EJEMPLOS_PENALES

    por_tipo = {e.tipo: e for e in EJEMPLOS_PENALES}
    return [{"id": f"U{i}", "etiqueta": por_tipo[t].etiqueta, "consulta": por_tipo[t].texto, "borrador": True,
             "revisado": False} for i, t in enumerate(EJEMPLOS_ESTUDIO, 1)]


def leer_casos(path: Path = CASOS_PATH) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def asignacion(n_participantes: int, casos: list[str]) -> pd.DataFrame:
    """Orden y método de cada participante. El método alterna caso a caso y de un participante al siguiente, así
    cada uno hace la mitad de los casos con cada método y, con un número par de participantes, cada caso se hace
    igual número de veces a mano y con la app. Los participantes pares recorren los casos en orden inverso: el
    método con el que se empieza también alterna (P01 a mano, P02 con la app...) y ningún caso va siempre primero."""
    filas = []
    for i in range(n_participantes):
        orden = list(enumerate(casos)) if i % 2 == 0 else list(enumerate(casos))[::-1]
        for k, caso in orden:
            filas.append({"participante": f"P{i + 1:02d}", "caso": caso, "metodo": METODOS[(k + i) % 2]})
    return pd.DataFrame(filas).reindex(columns=COLUMNAS)


def leer_registro(path: Path = REGISTRO_PATH) -> pd.DataFrame:
    """Solo las filas ya hechas (con minutos). Valida método, minutos > 0 y utilidad 1–5."""
    df = pd.read_csv(path, dtype={"participante": str, "caso": str, "metodo": str, "comentario": str})
    faltan = set(COLUMNAS) - set(df.columns)
    if faltan:
        raise ValueError(f"{path.name}: faltan columnas {sorted(faltan)}")
    df = df[df["minutos"].notna()].copy()
    for col in ("minutos", "precedentes_utiles", "utilidad"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    malos = df[~df["metodo"].isin(METODOS) | (df["minutos"] <= 0) | ~df["utilidad"].between(1, 5)]
    if len(malos):
        raise ValueError(f"{path.name}: filas inválidas (método manual/app, minutos > 0, utilidad 1–5): "
                         f"{', '.join(malos['participante'] + '/' + malos['caso'])}")
    return df


def resumir(df: pd.DataFrame, casos: list[dict]) -> Corrida:
    n_part = df["participante"].nunique()
    corrida = Corrida("estudio", parametros={"participantes": int(n_part), "registros": len(df)})
    if n_part < MIN_PARTICIPANTES:
        corrida.advertir(f"Estudio exploratorio: {n_part} participantes (menos de {MIN_PARTICIPANTES}). Los resultados "
                         "describen a estos participantes; no se generalizan.")
    if any(c.get("borrador") for c in casos):
        corrida.borrador = True
        corrida.advertir("Casos en borrador: un abogado debe revisarlos (borrador: false) antes del estudio.")
    por_metodo = []
    for metodo in METODOS:
        d = df[df["metodo"] == metodo]
        util = m.promedio_con_ic(d["utilidad"].tolist()) if len(d) else None
        por_metodo.append({
            "método": metodo, "registros": len(d),
            "mediana de minutos": float(d["minutos"].median()) if len(d) else None,
            "precedentes útiles por minuto": float(d["precedentes_utiles"].sum() / d["minutos"].sum()) if len(d) else None,
            "utilidad media (1–5)": util["valor"] if util else None,
            "IC 95 % utilidad": f"[{util['ic_inf']:.2f}, {util['ic_sup']:.2f}]" if util else "—"})
    corrida.tabla("Por método", "Mediana de minutos por tarea; precedentes útiles por minuto = total de precedentes "
                  "útiles / total de minutos; utilidad con IC bootstrap.", pd.DataFrame(por_metodo))
    med = df.groupby(["caso", "metodo"])["minutos"].median().unstack()
    pares = med.dropna(subset=[x for x in METODOS if x in med.columns]) if set(METODOS) <= set(med.columns) else med.iloc[0:0]
    if len(pares):
        d = m.bootstrap_diferencia_pareada(pares["manual"].tolist(), pares["app"].tolist())
        mde = m.efecto_minimo_detectable(pares["manual"].tolist(), pares["app"].tolist())
        corrida.tabla("Diferencia pareada por caso (manual − app, minutos)",
                      "Por caso, la mediana de minutos a mano menos la mediana con la app; positivo = la app ahorra "
                      "tiempo. IC bootstrap remuestreando casos. Efecto mínimo detectable: alfa 5 %, potencia 80 %.",
                      pd.DataFrame([{"casos": d["n"], "diferencia media": d["diferencia"],
                                     "IC 95 %": f"[{d['ic_inf']:.1f}, {d['ic_sup']:.1f}]",
                                     "IC excluye 0": "sí" if d["significativa"] else "no",
                                     "efecto mínimo detectable": mde if mde is not None else "—"}]))
        corrida.tabla("Por caso", "Mediana de minutos con cada método.",
                      pares.reset_index().assign(diferencia=lambda x: x["manual"] - x["app"]), diagnostico=True)
    else:
        corrida.advertir("Ningún caso tiene todavía registros con los dos métodos: diferencia pareada por medir.")
    corrida.extra["mediana_global_minutos"] = float(np.median(df["minutos"])) if len(df) else None
    return corrida
