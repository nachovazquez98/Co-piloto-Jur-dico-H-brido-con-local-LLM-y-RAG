"""Las métricas deben reproducir exactamente los ejemplos resueltos de la sección 5 del informe."""

import math

import pytest

from copiloto_juridico.evaluacion import metricas as m


def r3(x: float) -> float:
    return round(x, 3)


def test_ejemplo_ndcg_y_precision_del_informe():
    rels, ideales = [2, 0, 1, 2, 0], [2, 2, 1]
    assert r3(m.dcg_at_k(rels, 5)) == 3.361
    assert r3(m.dcg_at_k(ideales, 5)) == 3.762
    assert r3(m.ndcg_at_k(rels, 5, ideales)) == 0.894
    assert r3(m.precision_at_k(rels, 3)) == 0.667
    assert r3(m.precision_at_k(rels, 5)) == 0.600


def test_ejemplo_mrr_del_informe():
    # rangos [1, 3, ninguno, 2] -> (1 + 1/3 + 0 + 1/2) / 4
    rr = [m.reciprocal_rank(r) for r in ([1], [0, 0, 1], [0, 0], [0, 1])]
    assert r3(sum(rr) / len(rr)) == 0.458


def test_ejemplo_qc_del_informe():
    res = m.precision_recall_f1_binario(vp=54, fp=10, fn=6, vn=30)
    assert r3(res["precision"]) == 0.844
    assert r3(res["recall"]) == 0.900
    assert r3(res["f1"]) == 0.871
    assert r3(res["kappa"]) == 0.661
    assert r3(res["especificidad"]) == 0.750


def test_ejemplo_wilson_del_informe():
    lo, hi = m.wilson_ic(15, 18)
    assert (r3(lo), r3(hi)) == (0.608, 0.942)


def test_recuperacion_casos_borde():
    assert m.hit_at_k([0, 0, 1], 2) == 0.0 and m.hit_at_k([0, 0, 1], 3) == 1.0
    assert m.reciprocal_rank([0, 0, 2]) == pytest.approx(1 / 3)
    assert m.reciprocal_rank([0, 0]) == 0.0
    assert m.rango_primer_acierto([0, 1]) == 2
    assert m.precision_at_k([1], 5) == 0.2  # mostrar menos no se premia
    assert m.ndcg_at_k([0, 0], 5, [0]) == 0.0
    assert m.ndcg_at_k([2, 1], 5, [2, 1]) == 1.0


def test_clasificacion_multiclase():
    verdad = ["robo", "robo", "fraude", "lesiones"]
    pred = ["robo", "fraude", "fraude", "robo"]
    tabla = m.f1_por_clase(verdad, pred).set_index("clase")
    # robo: VP=1, FP=1, FN=1 -> P=R=F1=0.5; fraude: VP=1, FP=1, FN=0 -> P=0.5, R=1, F1=0.667; lesiones: 0
    assert tabla.loc["robo", "f1"] == pytest.approx(0.5)
    assert tabla.loc["fraude", "f1"] == pytest.approx(2 / 3)
    assert tabla.loc["lesiones", "f1"] == 0.0
    assert m.f1_macro(verdad, pred) == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    cm = m.matriz_confusion(verdad, pred)
    assert cm.loc["robo", "fraude"] == 1 and cm.loc["lesiones", "robo"] == 1 and int(cm.values.sum()) == 4


def test_kappa():
    assert m.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # Acuerdo igual al de azar
    assert m.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        m.cohen_kappa([1], [1, 0])


def test_bootstrap_reproducible_y_pareado():
    valores = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert m.bootstrap_ic(valores) == m.bootstrap_ic(valores)  # semilla fija
    lo, hi = m.bootstrap_ic(valores)
    assert 0 <= lo <= 0.7 <= hi <= 1
    d = m.bootstrap_diferencia_pareada([1] * 20, [0] * 20)
    assert d["diferencia"] == 1.0 and d["significativa"] and d["gana_a"] == 20
    d0 = m.bootstrap_diferencia_pareada([1, 0] * 10, [0, 1] * 10)
    assert d0["diferencia"] == 0.0 and not d0["significativa"]


def test_resumenes():
    r = m.proporcion_con_ic([1] * 15 + [0] * 3)
    assert r3(r["valor"]) == 0.833 and r["n"] == 18 and r3(r["ic_inf"]) == 0.608
    assert m.percentiles([1, 2, 3, 4, 100])["p50"] == 3.0
    assert not math.isnan(m.promedio_con_ic([0.5, 1.0])["valor"])
