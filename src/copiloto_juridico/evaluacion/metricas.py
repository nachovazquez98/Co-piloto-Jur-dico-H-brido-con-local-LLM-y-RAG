"""Métricas de evaluación (funciones puras, sin dependencias del modelo ni de la base).

Convenciones usadas en todo el módulo:

- Una *lista de relevancias* es la relevancia de cada documento devuelto, **en el orden** en que el sistema los
  mostró: `[2, 0, 1]` significa que el primero es muy relevante, el segundo irrelevante y el tercero
  parcialmente relevante (escala 0/1/2 del Anexo B del informe).
- Los rangos empiezan en 1 (el primer resultado tiene rango 1). `None` significa "no apareció".
- Un documento es *relevante* para las métricas binarias si su relevancia es >= `umbral` (por defecto 1).

Las fórmulas siguen la sección 5 del informe y Manning, Raghavan y Schütze (2008).
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist
from typing import Callable, Hashable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------------------
# Recuperación
# --------------------------------------------------------------------------------------------------------------


def precision_at_k(relevancias: Sequence[float], k: int, umbral: float = 1) -> float:
    """Precision@k: fracción de los k primeros resultados que son relevantes.

    Responde "de lo que le muestro al usuario, ¿cuánto le sirve?". Si el sistema devolvió menos de k
    documentos, los faltantes cuentan como no relevantes (el denominador siempre es k), que es la convención
    estándar: mostrar menos no debe premiarse.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")
    return sum(1 for r in list(relevancias)[:k] if r >= umbral) / k


def hit_at_k(relevancias: Sequence[float], k: int, umbral: float = 1) -> float:
    """Hit@k: 1 si al menos un documento relevante aparece en los k primeros, 0 si no.

    Promediado entre consultas, es la fracción de consultas en las que el usuario ve al menos un acierto.
    """
    return 1.0 if any(r >= umbral for r in list(relevancias)[:k]) else 0.0


def rango_primer_acierto(relevancias: Sequence[float], umbral: float = 1) -> int | None:
    """Posición (empezando en 1) del primer documento relevante, o None si ninguno lo es."""
    for i, r in enumerate(relevancias, 1):
        if r >= umbral:
            return i
    return None


def reciprocal_rank(relevancias: Sequence[float], umbral: float = 1) -> float:
    """Rango recíproco: 1 / posición del primer acierto (0 si no hay acierto).

    Premia mucho que el acierto esté arriba: 1.º lugar = 1, 2.º = 0.5, 3.º = 0.33, 10.º = 0.1.
    """
    rango = rango_primer_acierto(relevancias, umbral)
    return 1.0 / rango if rango else 0.0


def dcg_at_k(relevancias: Sequence[float], k: int) -> float:
    """DCG@k (Discounted Cumulative Gain) con ganancia lineal: Σ_{i=1..k} rel_i / log2(i + 1).

    Suma la relevancia de cada resultado, descontada por su posición: un documento muy relevante en el lugar 1
    vale su relevancia completa; en el lugar 3 vale la mitad (log2(4) = 2). Así se premia el buen *orden*.
    """
    return sum(r / math.log2(i + 1) for i, r in enumerate(list(relevancias)[:k], 1))


def ndcg_at_k(relevancias: Sequence[float], k: int, ideales: Sequence[float] | None = None) -> float:
    """nDCG@k: DCG@k dividido entre el DCG del mejor orden posible (DCG ideal), en [0, 1].

    `ideales` son las relevancias de *todos* los documentos relevantes conocidos para la consulta (no solo los
    devueltos). Se ordenan de mayor a menor para obtener el orden ideal. Si no se dan, se usa el mejor orden de
    los propios resultados (nDCG "local", más optimista).

    Ejemplo del informe: relevancias [2, 0, 1, 2, 0] e ideales [2, 2, 1] -> 3.361 / 3.762 = 0.894.
    1.0 = orden perfecto; el 11 % que falta se debe al documento irrelevante en el segundo lugar.
    """
    ideal = sorted(ideales if ideales is not None else relevancias, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(relevancias, k) / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------------------------------------------
# Clasificación multiclase
# --------------------------------------------------------------------------------------------------------------


def matriz_confusion(verdaderos: Sequence[Hashable], predichos: Sequence[Hashable],
                     etiquetas: Sequence[Hashable] | None = None) -> pd.DataFrame:
    """Matriz de confusión: filas = clase verdadera, columnas = clase predicha, celdas = conteos.

    La diagonal son los aciertos; una celda fuera de la diagonal (fila "robo", columna "despojo") dice cuántas
    veces el sistema confundió un delito con otro.
    """
    if etiquetas is None:
        etiquetas = sorted(set(verdaderos) | set(predichos), key=str)
    m = pd.DataFrame(0, index=pd.Index(list(etiquetas), name="verdadero"),
                     columns=pd.Index(list(etiquetas), name="predicho"))
    for v, p in zip(verdaderos, predichos):
        if v in m.index and p in m.columns:
            m.loc[v, p] += 1
    return m


def f1_por_clase(verdaderos: Sequence[Hashable], predichos: Sequence[Hashable],
                 etiquetas: Sequence[Hashable] | None = None) -> pd.DataFrame:
    """Precisión, recall, F1 y soporte de cada clase (tabla). Una clase sin predicciones tiene precisión 0."""
    if etiquetas is None:
        etiquetas = sorted(set(verdaderos) | set(predichos), key=str)
    filas = []
    for c in etiquetas:
        vp = sum(1 for v, p in zip(verdaderos, predichos) if v == c and p == c)
        fp = sum(1 for v, p in zip(verdaderos, predichos) if v != c and p == c)
        fn = sum(1 for v, p in zip(verdaderos, predichos) if v == c and p != c)
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        filas.append({"clase": c, "precision": prec, "recall": rec, "f1": f1, "soporte": vp + fn})
    return pd.DataFrame(filas)


def f1_macro(verdaderos: Sequence[Hashable], predichos: Sequence[Hashable],
             etiquetas: Sequence[Hashable] | None = None) -> float:
    """F1 macro: promedio simple del F1 de cada clase, F1_c = 2·P_c·R_c / (P_c + R_c).

    Todas las clases pesan igual, sin importar cuántos ejemplos tengan: con clases desbalanceadas evita que el
    resultado lo dominen los delitos frecuentes. Por defecto las clases son la unión de verdaderas y predichas
    (misma convención que scikit-learn): predecir una clase inexistente cuenta como F1 = 0 para esa clase.
    """
    tabla = f1_por_clase(verdaderos, predichos, etiquetas)
    return float(tabla["f1"].mean()) if len(tabla) else 0.0


# --------------------------------------------------------------------------------------------------------------
# Clasificación binaria (QC_1, QC_2) y acuerdo
# --------------------------------------------------------------------------------------------------------------


def precision_recall_f1_binario(vp: int, fp: int, fn: int, vn: int) -> dict[str, float]:
    """Métricas de un clasificador binario a partir de su matriz de confusión.

    En los controles de calidad, "positivo" = el control deja pasar el documento (VEREDICTO: SI) y la verdad la
    pone el anotador humano:
    - precisión = VP / (VP + FP): de lo que el control deja pasar, cuánto es realmente relevante;
    - recall (sensibilidad) = VP / (VP + FN): de lo relevante, cuánto conserva (lo demás es descarte injusto);
    - especificidad = VN / (VN + FP): de lo irrelevante, cuánto elimina;
    - F1 = media armónica de precisión y recall;
    - exactitud = (VP + VN) / total;
    - kappa de Cohen (ver `kappa_desde_matriz`).
    Ejemplo del informe: VP=54, FP=10, FN=6, VN=30 -> precisión 0.844, recall 0.900, F1 0.871, kappa 0.661.
    """
    total = vp + fp + fn + vn
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "especificidad": vn / (vn + fp) if vn + fp else 0.0,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "exactitud": (vp + vn) / total if total else 0.0,
        "kappa": kappa_desde_matriz([[vp, fp], [fn, vn]]),
        "n": total,
    }


def kappa_desde_matriz(matriz: Sequence[Sequence[int]] | np.ndarray) -> float:
    """Kappa de Cohen desde una matriz de acuerdo cuadrada (filas = evaluador A, columnas = evaluador B).

    κ = (p_o − p_e) / (1 − p_e), donde p_o es el acuerdo observado (la diagonal) y p_e el acuerdo que se
    esperaría por azar si cada evaluador respondiera al azar con sus propias proporciones. 0 = acuerdo de azar,
    1 = acuerdo perfecto; 0.61–0.80 se suele leer como "sustancial" (Landis y Koch, 1977).
    """
    m = np.asarray(matriz, dtype=float)
    n = m.sum()
    if n == 0:
        return 0.0
    p_o = np.trace(m) / n
    p_e = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / n ** 2
    return float((p_o - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def cohen_kappa(anotador_a: Sequence[Hashable], anotador_b: Sequence[Hashable]) -> float:
    """Kappa de Cohen entre dos listas de etiquetas emparejadas (p. ej. dos anotadores, o QC_1 vs. humano).

    Mide el acuerdo descontando el que ocurriría por azar. Se usa para el acuerdo entre anotadores del
    conjunto B (si es < 0.6, se revisa la guía de anotación) y para comparar los QC con el juicio humano.
    """
    if len(anotador_a) != len(anotador_b):
        raise ValueError("Las dos listas deben tener la misma longitud")
    etiquetas = sorted(set(anotador_a) | set(anotador_b), key=str)
    pos = {e: i for i, e in enumerate(etiquetas)}
    m = np.zeros((len(etiquetas), len(etiquetas)))
    for a, b in zip(anotador_a, anotador_b):
        m[pos[a], pos[b]] += 1
    return kappa_desde_matriz(m)


# --------------------------------------------------------------------------------------------------------------
# Incertidumbre
# --------------------------------------------------------------------------------------------------------------


def _z(confianza: float) -> float:
    return NormalDist().inv_cdf(0.5 + confianza / 2)


def wilson_ic(exitos: int, n: int, confianza: float = 0.95) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción (p. ej. Hit@3 = aciertos / consultas).

    Con pocas consultas una proporción puede variar mucho: 15/18 = 0.83, pero el intervalo al 95 % es
    [0.608, 0.942]. Se prefiere a "p ± 1.96·√(p(1−p)/n)" porque no se sale de [0, 1] y funciona bien con n
    pequeño o proporciones cercanas a 0 o 1.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = _z(confianza)
    p = exitos / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    mitad = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centro - mitad), min(1.0, centro + mitad))


def bootstrap_ic(valores: Sequence[float], estadistico: Callable[[np.ndarray], float] = np.mean,
                 n_remuestreos: int = 1000, semilla: int = 42, confianza: float = 0.95) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap de percentiles (para MRR, nDCG y cualquier promedio).

    Se remuestrean las consultas con reemplazo 1,000 veces, se recalcula el estadístico en cada remuestreo y se
    toman los percentiles 2.5 y 97.5. Idea: "si hubiera escrito otras consultas parecidas, ¿cuánto cambiaría el
    resultado?". La semilla fija hace que el intervalo sea reproducible.
    """
    x = np.asarray(valores, dtype=float)
    if len(x) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(x), size=(n_remuestreos, len(x)))
    stats = np.array([estadistico(x[i]) for i in idx])
    alfa = (1 - confianza) / 2
    return (float(np.quantile(stats, alfa)), float(np.quantile(stats, 1 - alfa)))


def bootstrap_diferencia_pareada(a: Sequence[float], b: Sequence[float], n_remuestreos: int = 1000,
                                 semilla: int = 42, confianza: float = 0.95) -> dict[str, float | bool]:
    """Diferencia pareada media(a − b) con intervalo bootstrap, para comparar dos variantes.

    `a` y `b` son el resultado de cada variante **en las mismas consultas** (p. ej. Hit@3 por consulta con y
    sin Jaccard). Al remuestrear se conservan los pares, así se descuenta que unas consultas son más difíciles
    que otras. La mejora se acepta solo si el intervalo no incluye el cero (`significativa`).
    """
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(xa) != len(xb):
        raise ValueError("Las variantes deben evaluarse sobre las mismas consultas")
    dif = xa - xb
    lo, hi = bootstrap_ic(dif, np.mean, n_remuestreos, semilla, confianza)
    return {"diferencia": float(dif.mean()) if len(dif) else 0.0, "ic_inf": lo, "ic_sup": hi,
            "significativa": bool(lo > 0 or hi < 0), "n": int(len(dif)),
            "gana_a": int((dif > 0).sum()), "gana_b": int((dif < 0).sum()), "empates": int((dif == 0).sum())}


def proporcion_con_ic(aciertos: Sequence[float], confianza: float = 0.95) -> dict[str, float]:
    """Resumen de una métrica 0/1 por consulta: valor, IC de Wilson y n."""
    x = [float(v) for v in aciertos]
    n, exitos = len(x), int(round(sum(x)))
    lo, hi = wilson_ic(exitos, n, confianza)
    return {"valor": exitos / n if n else 0.0, "ic_inf": lo, "ic_sup": hi, "n": n}


def promedio_con_ic(valores: Sequence[float], semilla: int = 42, confianza: float = 0.95) -> dict[str, float]:
    """Resumen de una métrica continua por consulta (RR, nDCG): media, IC bootstrap y n."""
    x = [float(v) for v in valores]
    lo, hi = bootstrap_ic(x, semilla=semilla, confianza=confianza)
    return {"valor": float(np.mean(x)) if x else 0.0, "ic_inf": lo, "ic_sup": hi, "n": len(x)}


def percentiles(valores: Sequence[float], qs: Sequence[float] = (50, 95)) -> dict[str, float]:
    """Percentiles de una lista (latencias): p50 = experiencia típica, p95 = la peor razonable."""
    x = np.asarray(valores, dtype=float)
    return {f"p{int(q)}": float(np.percentile(x, q)) if len(x) else 0.0 for q in qs}


def conteo(etiquetas: Sequence[Hashable]) -> dict:
    return dict(Counter(etiquetas))
