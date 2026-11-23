"""Pruebas del módulo de evaluación sin GPU, sin Ollama y sin la base vectorial (datos pequeños)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from copiloto_juridico.clasificador import PesosClasificador, puntuar
from copiloto_juridico.evaluacion import conjuntos as cj
from copiloto_juridico.evaluacion import experimentos as ex
from copiloto_juridico.evaluacion import familias as fam
from copiloto_juridico.fichas import (datos_penales, extraer_articulos,
                                      normalizar_sentido, parsear_campos)

# --- Familias -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tipo, familia", [
    ("penal/robo-calificado", "robo"),
    ("penal/tentativa-de-robo-calificado", "robo"),
    ("penal/robo-de-infante", "sustraccion-de-menores"),
    ("penal/violacion", "delitos-sexuales"),
    ("penal/violacion-de-deposito", "fraude-y-abuso-de-confianza"),
    ("penal/homicidio-a-titulo-de-culpa", "delitos-culposos"),
    ("penal/lesiones-y-dano-en-las-cosas-ambos-a-titulo-de-culpa", "delitos-culposos"),
    ("penal/homicidio-calificado-y-robo-calificado", "homicidio"),
    ("penal/despojo-de-inmuebles-y-aguas", "despojo"),
    ("penal/contra-la-salud-en-su-modalidad-de-narcomenudeo", "contra-la-salud"),
    ("penal/crueldad-animal", "otros"),
    ("civil/accion-de-divorcio", "civil/accion-de-divorcio"),
])
def test_asignar_familia(tipo, familia):
    assert fam.asignar_familia(tipo).familia == familia


def test_familia_marca_compuestos_y_no_falsos_positivos():
    a = fam.asignar_familia("penal/homicidio-calificado-y-robo-calificado")
    assert a.revisar and "robo" in a.nota
    assert "delitos-sexuales" not in fam.asignar_familia("penal/violacion-de-deposito").nota
    assert not fam.asignar_familia("penal/despojo-de-inmuebles-y-aguas").revisar


def test_csv_revisado_manda(tmp_path):
    path = tmp_path / "familias.csv"
    fam.exportar_csv({"penal/robo": 3, "penal/feminicidio": 2}, path)
    with pytest.raises(FileExistsError):
        fam.exportar_csv({"penal/robo": 3}, path)
    df = pd.read_csv(path)
    df.loc[df.tipo == "penal/feminicidio", "familia"] = "homicidio"  # decisión del abogado
    df.to_csv(path, index=False)
    mapa = fam.cargar_familias(["penal/robo", "penal/feminicidio", "penal/fraude"], path)
    assert mapa == {"penal/robo": "robo", "penal/feminicidio": "homicidio", "penal/fraude": "fraude-y-abuso-de-confianza"}


# --- Conjuntos ------------------------------------------------------------------------------------------------


def _corpus():
    filas = []
    for t, n in (("penal/robo-calificado", 12), ("penal/robo-simple", 3), ("penal/fraude", 11), ("penal/lesiones", 2)):
        for i in range(n):
            filas.append({"id": f"{t.split('/')[1]}-{i}.pdf", "materia": "penal", "tipo": t, "fecha": "2025-01-01",
                          "texto": f"vistos para resolver el toca {t} numero {i} " * 20})
    return pd.DataFrame(filas)


def test_validacion_lista_todos_los_problemas(tmp_path):
    corpus = _corpus()
    path = tmp_path / "c.jsonl"
    filas = [
        {"id": "1", "consulta": "robo de celular", "doc_id_fuente": "robo-calificado-0.pdf",
         "tipo": "penal/robo-calificado", "familia": "robo", "particion": "desarrollo", "sin_respuesta": False},
        {"id": "1", "consulta": "", "doc_id_fuente": "no-existe.pdf", "tipo": "penal/robo-calificado",
         "familia": "robo", "particion": "otra", "sin_respuesta": False},
        {"id": "3", "consulta": "x", "doc_id_fuente": "fraude-0.pdf", "tipo": "penal/fraude", "familia": "robo",
         "particion": "prueba", "sin_respuesta": True},
    ]
    path.write_text("\n".join(json.dumps(f) for f in filas))
    familias = {t: fam.asignar_familia(t).familia for t in corpus["tipo"].unique()}
    with pytest.raises(cj.ErrorValidacion) as e:
        cj.cargar_consultas(path, corpus, familias)
    msg = str(e.value)
    for esperado in ("id duplicado", "vacía", "partición", "no está en el corpus", "sin respuesta", "familia"):
        assert esperado in msg
    ok = cj.cargar_consultas(_escribir(tmp_path, filas[:1]), corpus, familias)
    assert ok[0].documento_conocido


def _escribir(tmp_path, filas):
    p = tmp_path / "ok.jsonl"
    p.write_text("\n".join(json.dumps(f) for f in filas))
    return p


def test_juicios_se_validan(tmp_path):
    p = tmp_path / "j.csv"
    p.write_text("consulta_id,doc_id,relevancia,anotador\nB1,d1,2,ana\nB1,d2,0,ana\nB1,d1,2,luis\nB1,d2,1,luis\n")
    j = cj.cargar_juicios(p, {"B1"})
    assert cj.relevancia_por_par(j, "ana") == {"B1": {"d1": 2, "d2": 0}}
    p.write_text("consulta_id,doc_id,relevancia,anotador\nB1,d1,3,ana\n")
    with pytest.raises(cj.ErrorValidacion):
        cj.cargar_juicios(p)


def test_fuga_ngramas():
    fuente = "El imputado sustrajo con violencia el teléfono celular de la víctima en la vía pública de noche."
    copia = "Mi cliente sustrajo con violencia el telefono celular de la victima, según el MP."
    propia = "Le quitaron el móvil a una persona en la calle usando la fuerza; quiero apelar la condena."
    assert cj.ngramas_compartidos(copia, fuente)  # 7 palabras seguidas iguales (sin acentos)
    assert cj.ngramas_compartidos(propia, fuente) == []
    c = cj.Consulta("x", "es un robo calificado en la calle", "d", "penal/robo-calificado", "robo", "prueba")
    assert cj.nombre_tipo_en_consulta(c)


# --- Fichas ---------------------------------------------------------------------------------------------------

FICHA = """- **Tribunal:** Primera Sala Penal, 12 de marzo de 2025
- **Etapa Procesal:** Auto de vinculación a proceso
- **Delito(s) Principal(es):** Robo calificado (art. 236)
- **Acto Impugnado:** Auto de vinculación a proceso
- **Agravios Planteados:**
  1. Falta de valoración del testimonio.
- **Sentido del Fallo:** CONFIRMA
- **Fundamento Legal:** artículos 233, 234 y 236 del Código Penal; numeral 316 del CNPP; art. 168 bis
"""


def test_parsear_ficha():
    c = parsear_campos(FICHA)
    assert c["etapa procesal"] == "Auto de vinculación a proceso"
    assert "testimonio" in c["agravios planteados"]
    d = datos_penales(FICHA)
    assert d["sentido_fallo"] == "CONFIRMA" and d["acto_impugnado"].startswith("Auto")
    assert extraer_articulos(FICHA) == {"233", "234", "236", "316", "168bis"}


def test_sentido_y_coincidencias():
    assert normalizar_sentido("Se MODIFICA la sentencia apelada en cuanto a la pena") == "MODIFICA"
    assert normalizar_sentido("Se revocó el auto") == "REVOCA"
    assert normalizar_sentido("Se ordena la reposición del procedimiento") == "REPONE"
    assert normalizar_sentido("[DATO NO DISPONIBLE]") is None
    assert normalizar_sentido("Absuelve") == "OTRO"


# --- Experimentos (funciones puras) ---------------------------------------------------------------------------


def test_resumen_categorias_y_familia():
    familias = {"penal/robo-calificado": "robo", "penal/robo-simple": "robo", "penal/fraude": "fraude"}
    r = ex.resumen_categorias("penal/robo-simple", "robo", ["penal/fraude", "penal/robo-calificado", "penal/x"],
                              lambda c: [c], familias)
    assert r["hit3_tipo"] == 0.0 and r["hit3_familia"] == 1.0 and r["top1_familia"] == 0.0
    # A nivel familia una categoría cubre varios tipos: top1_tipo no aplica
    r = ex.resumen_categorias("penal/robo-simple", "robo", ["penal/familia/robo"],
                              lambda c: ["penal/robo-calificado", "penal/robo-simple"], familias)
    assert r["hit3_tipo"] == 1.0 and np.isnan(r["top1_tipo"]) and r["top1_familia"] == 1.0


def test_metricas_documento_y_metadatos():
    r = ex.metricas_documento(["a", "b", "fuente", "c"], "fuente")
    assert r["hit1"] == 0 and r["hit3"] == 1 and r["rr"] == pytest.approx(1 / 3)
    doc_tipo = {"a": "penal/robo", "b": "penal/robo-simple", "c": "penal/fraude"}
    familias = {"penal/robo": "robo", "penal/robo-simple": "robo", "penal/fraude": "fraude"}
    assert ex.relevancias_metadatos(["a", "b", "c"], "penal/robo", "robo", doc_tipo, familias) == [2, 1, 0]
    assert ex.ideales_metadatos("penal/robo", "robo", {"penal/robo": 2, "penal/robo-simple": 1, "penal/fraude": 5},
                                familias) == [2, 2, 1]
    g = ex.metricas_graduadas(["a", "z"], {"a": 2, "q": 1})
    assert g["cobertura"] == 0.2 and 0 < g["ndcg5"] < 1


def test_knn_mayoritario():
    v = [SimpleNamespace(tipo=t, similitud=s) for t, s in
         (("A", 0.9), ("B", 0.89), ("B", 0.8), ("A", 0.7), ("C", 0.95))]
    assert ex.ranking_knn(v) == ["B", "A", "C"]  # empate 2-2: B gana por similitud acumulada (1.69 > 1.60)


def _comp():
    return pd.DataFrame({"tipo": ["t1", "t2", "t3"], "coseno": [1.0, 0.0, 0.5], "jaccard": [0.0, 1.0, 0.0],
                         "kem": [0.0, 1.0, 0.0], "concision": [0.5, 0.5, 0.5], "voto": [0.0, 0.0, 1.0],
                         "penal_longitud": [1.0, 1.0, 0.9], "penal_tokens": [1.0, 0.7, 1.0]})


def test_ablacion_y_penalizaciones():
    comp, pesos = _comp(), PesosClasificador(0.20, 0.20, 0.20, 0.05, 0.35, True, True)  # pesos de v10
    s = puntuar(comp, pesos)
    assert s[1] == pytest.approx((0.2 + 0.2 + 0.025) * 0.7)
    assert puntuar(comp, pesos.sin("penal_tokens"))[1] == pytest.approx(0.425)
    assert ex.ranking_categorias(comp, pesos.sin("voto"))[0] == "t2"
    with pytest.raises(ValueError):
        pesos.sin("inventado")


def test_malla_pesos_y_evaluacion_vectorizada():
    W = ex.malla_pesos(0.05)
    assert W.shape == (10626, 5) and np.allclose(W.sum(axis=1), 1) and (W >= 0).all()
    comp = _comp()
    caso = ex.caso_malla(comp, "t2", "f2", {"t1": "f1", "t2": "f2", "t3": "f3"}, PesosClasificador.desde_config())
    res = ex.evaluar_malla([caso], np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0.0]]), k=1)
    assert res["top1_tipo"].tolist() == [1.0, 0.0]
    # Debe coincidir con el ranking no vectorizado
    pesos = PesosClasificador(*W[1234], True, True)
    esperado = float(ex.ranking_categorias(comp, pesos)[0] == "t2")
    assert ex.evaluar_malla([caso], W[1234:1235], k=1)["top1_tipo"][0] == esperado


def test_resumen_binario_falla_abierta():
    r = ex.resumen_binario([True, True, False, False], [True, None, None, False])
    assert (r["VP"], r["FP"], r["FN"], r["VN"]) == (2, 1, 0, 1)
    assert r["tasa_no_interpretables"] == 0.5


def test_auc_y_senales():
    assert ex.auc_roc([3, 4], [1, 2]) == 1.0 and ex.auc_roc([1], [1]) == 0.5
    t = pd.DataFrame({"score_híbrido": [0.6, 0.4], "coseno": [0.8, 0.7], "kem": [0.0, 1.0]})
    assert ex.senales_sin_respuesta(t) == {"score_hibrido": 0.6, "coseno": 0.8, "kem_max": 1.0}


class _LLMFalso:
    def __init__(self):
        self.settings = SimpleNamespace(provider="x", model="m", temperature=0.0, max_tokens=10, num_ctx=100,
                                        seed=42, max_context_chars=1000)
        self.n = 0

    def generate(self, prompt):
        self.n += 1
        return f"respuesta a {prompt}"


def test_cache_llm_persiste(tmp_path):
    base = _LLMFalso()
    c1 = ex.LLMConCache(base, tmp_path / "llm.jsonl")
    assert c1.generate("hola") == c1.generate("hola") and base.n == 1
    c2 = ex.LLMConCache(base, tmp_path / "llm.jsonl")
    assert c2.generate("hola") == "respuesta a hola" and base.n == 1 and c2.aciertos == 1


def test_conceptos_en_cache_se_invalidan_si_cambia_la_consulta():
    c = cj.Consulta("A1", "texto original", None, None, None, "prueba")
    cache = {"A1": {"clave": ex.clave_conceptos(c), "conceptos": "robo"}}
    assert ex.conceptos_de([c], cache) == ({"A1": "robo"}, [])
    c.consulta = "texto editado"
    assert ex.conceptos_de([c], cache) == ({"A1": "texto editado"}, ["A1"])


def test_reporte_guarda_json_y_md(tmp_path):
    from copiloto_juridico.evaluacion.reporte import Corrida, ultimo_resultado

    cor = Corrida("prueba", parametros={"particion": "desarrollo"}, borrador=True)
    cor.tabla("Tabla", "desc", pd.DataFrame([{"variante": "x", "Hit@3": 0.5, "n": np.int64(3)}]))
    pj, pm = cor.guardar(tmp_path)
    datos = json.loads(pj.read_text())
    assert datos["borrador"] and datos["tablas"]["Tabla"]["filas"][0]["n"] == 3
    assert "configuracion" in datos and "corpus_fingerprint" in datos
    assert "BORRADOR" in pm.read_text()
    assert ultimo_resultado("prueba", tmp_path)["_archivo"] == pj.name


# Fragmentos reales de resolutivos del corpus (texto tal como lo deja la extracción del PDF).
RESOLUTIVOS_REALES = [
    ("2025-06-05-110514_penal_violencia-intrafamiliar.pdf", "CONFIRMA",
     "resuelve en base a las siguientes p r o p o s i c i o n e s primera. - se confirma la resolución pronunciada "
     "el 06 seis de mayo de 2019 dos mil diecinueve, por el juez déc"),
    ("2025-10-10-135730_penal_lesiones-a-titulo-de-culpa.pdf", "MODIFICA",
     "r e s u e l v e primera. - se modifica el auto de vinculación a proceso de fecha 24 veinticuatro de enero de "
     "2025 dos mil veint"),
    ("2025-04-10-132125_penal_fraude-especifico.pdf", "REVOCA",
     "r e s u e l v e  primera. por los motivos y fundamentos cuerpo de la presente, se revoca el auto de vinculación "
     "a proceso, pronunciado el día 22 veintidós de abril de 2024 dos m"),
    ("2025-12-08-113827_penal_homicidio-culposo.pdf", "REPONE",
     "r e s u e l v e  primer o. se ordena la reposición del procedimiento de primera insta ncia, a fin de que se "
     "desahogue nuevamente la totali"),
]


@pytest.mark.parametrize("doc, esperado, texto", RESOLUTIVOS_REALES)
def test_sentido_por_reglas_con_resolutivos_reales(doc, esperado, texto):
    from copiloto_juridico.fichas import sentido_por_reglas

    assert sentido_por_reglas(texto) == esperado


def test_sentido_por_reglas_casos_dificiles():
    from copiloto_juridico.fichas import sentido_por_reglas

    # Revoca y ordena reponer: manda la reposición
    assert sentido_por_reglas("se resuelve primera. se revoca la sentencia y se ordena la reposición del "
                              "procedimiento") == "REPONE"
    # Argumento en el cuerpo (sin sección de resolutivos): no se inventa un sentido
    assert sentido_por_reglas("en primera instancia no es suficiente para reponer el procedimiento ni para "
                              "confirmar lo resuelto") is None
    # Anexo de datos testados al final no impide encontrar los resolutivos
    anexo = " no.214 eliminadoelnombrecompleto en 1 renglones por ser un dato identificativo lgppicr." * 40
    assert sentido_por_reglas("x " * 300 + "r e s u e l v e primero. se confirma el auto" + anexo) == "CONFIRMA"
    # v11.2: "lo procedente es ..." y el verbo sin "se"; este último solo si no hay otro (la sección también narra)
    assert sentido_por_reglas("puntos resolutivos primero. por los razonamientos de este fallo, lo procedente es "
                              "revocar el auto de no vinculación a proceso") == "REVOCA"
    assert sentido_por_reglas("proposiciones primera. por los motivos y fundamentos revoca lo resuelto en la "
                              "audiencia") == "REVOCA"
    assert sentido_por_reglas("resuelve primero. el tribunal colegiado confirma el amparo indirecto; segundo. se "
                              "revoca la resolución apelada") == "REVOCA"
    # "Se deja insubsistente" no basta: puede ser la sentencia de la propia sala o una competencia
    assert sentido_por_reglas("resuelve primera. se deja insubsistente la sentencia dictada por este ad quem") is None


def test_articulos_con_ordinal_y_no_encontrados():
    from copiloto_juridico.fichas import extraer_articulos, verificar_articulos

    assert extraer_articulos("de conformidad con el ordinal 82 fracción i, y los ordinales 261, 263") == {"82", "261", "263"}
    assert verificar_articulos("Art. 233 y artículo 161", "... el artículo 233 del código ...") == ["161"]


def test_articulos_en_sentencia_lectura_tolerante():
    from copiloto_juridico.fichas import articulos_en_sentencia, verificar_articulos

    texto = ("con apoyo en lo dispuesto por los artículos 20 fracción i, 133 fracción iii, 461, 467 fracción vii, 475, "
             "478 y 479 del código nacional; en relación al 219, fracción i; los arábigos 461, 473; los numeral es "
             "316, 317; previsto en el 250 del código penal; la audiencia del 25 de enero; 16 constitucional")
    s = articulos_en_sentencia(texto)
    assert {"20", "133", "461", "467", "475", "478", "479", "219", "473", "316", "317", "250", "16"} <= s
    assert "25" not in s  # una fecha no es un artículo
    assert verificar_articulos("Artículos 467 y 999", texto) == ["999"]


def test_ficha_penal_desde_json():
    from copiloto_juridico.fichas import FichaPenal, familia_del_delito

    f = FichaPenal.desde_json('Aquí está: {"delitos": "Robo calificado", "resumen": "Se confirma.", '
                              '"sentido_fallo": "confirma", "agravios": [{"agravio": "A", "calificacion": "infundados"}],'
                              ' "fundamento_legal": "artículo 233; artículo 161"}')
    assert f.sentido_fallo == "CONFIRMA" and f.agravios[0]["calificacion"] == "INFUNDADO"
    assert f.fundamento_legal == ["artículo 233", "artículo 161"] and f.tribunal == "[DATO NO DISPONIBLE]"
    assert "no encontrado en la sentencia: art. 161" in f.informe_markdown(["161"])
    assert "- **Sentido del Fallo:** CONFIRMA" in f.campos_markdown()
    for malo in ("sin json", '{"delitos": "x"}', '{"delitos": "x", "resumen": "y", "sentido_fallo": "quizá"}'):
        with pytest.raises(ValueError):
            FichaPenal.desde_json(malo)
    assert familia_del_delito("Delitos contra la salud en su modalidad de narcomenudeo") == "contra-la-salud"
    assert familia_del_delito("Tocamientos a un menor") == "otros"


def test_borradores_reintentan_si_copian_y_quedan_sin_revisar():
    from copiloto_juridico.evaluacion.anotacion import redactar_borradores

    fuente = "el imputado sustrajo con violencia el teléfono celular de la víctima en la vía pública de noche"

    class LLM:
        def __init__(self):
            self.n = 0

        def generate(self, prompt):
            self.n += 1
            if "sin respuesta" in prompt or "Escribe en 2 a 4 frases" in prompt:
                return "Mi cliente adolescente fue internado; quiero apelar la medida."
            return fuente if self.n == 1 else "Le quitaron el móvil a alguien en la calle; quiero apelar."

    plantilla = [cj.Consulta("A1", "", "d1", "penal/robo-calificado", "robo", "prueba"),
                 cj.Consulta("S1", "", None, None, None, "prueba", sin_respuesta=True, extra={"tema_sugerido": "x"})]
    llm = LLM()
    out = redactar_borradores(plantilla, pd.Series({"d1": fuente}), llm, log=lambda m: None)
    assert out[0].extra["intentos"] == 2 and "fuga" not in out[0].extra
    assert all(c.borrador and not c.revisado for c in out) and out[1].consulta.startswith("Mi cliente")
    with pytest.raises(cj.ErrorValidacion, match="sin revisar"):
        cj.exigir_revisadas(out)
    for c in out:
        c.revisado = True
    cj.exigir_revisadas(out)


def test_resultado_para_defensa():
    """Quién ganó se deduce con reglas (apelante + sentido), no con el LLM."""
    from copiloto_juridico.fichas import resultado_para_defensa as r

    assert r("Defensa del imputado", "CONFIRMA") == "desfavorable"
    assert r("el sentenciado", "REVOCA") == "favorable"
    assert r("Ministerio Público", "CONFIRMA") == "favorable"
    assert r("Asesor jurídico de la víctima", "MODIFICA") == "parcialmente desfavorable"
    assert r("el imputado y el fiscal", "CONFIRMA") is None  # apelaron ambos
    assert r("Defensa", "OTRO") is None and r("[DATO NO DISPONIBLE]", "CONFIRMA") is None


def test_parte_favorecida():
    """La misma regla, dicha como la parte que obtuvo la resolución (sirve a los dos perfiles de usuario)."""
    from copiloto_juridico.fichas import parte_favorecida as f

    assert f("Defensa del imputado", "CONFIRMA") == "la acusación (Ministerio Público o víctima)"
    assert f("el sentenciado", "REVOCA") == "el acusado"
    assert f("Asesor jurídico de la víctima", "MODIFICA") == "en parte, la acusación (Ministerio Público o víctima)"
    assert f("Defensa", "REPONE") == "el acusado; se repone el procedimiento"
    assert f("el imputado y el fiscal", "CONFIRMA") is None


def test_detalles_identificables():
    from copiloto_juridico.evaluacion.conjuntos import detalles_identificables as d

    assert d("en la audiencia del 28 de junio de 2024 lo condenaron a tres años de prisión") == [
        "28 de junio de 2024", "tres años"]
    assert d("la carpeta 52192024") == ["52192024"]
    assert d("Mi cliente fue detenido sin orden y apelo la vinculación a proceso") == []
    assert d("abusó de una menor de 14 años") == []
    assert d("le impusieron una pena de 8 años de prisión") == ["8 años"]
