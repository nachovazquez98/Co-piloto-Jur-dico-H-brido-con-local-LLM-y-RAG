"""Referencias de la evaluación automática (v11.2): lo que haría el anotador humano, sin anotador humano.

- **Delito:** la categoría que el STJ asignó a la sentencia en su portal (externa al sistema) → familia.
- **Acto impugnado, apelante y sentido del fallo:** tres lectores independientes de la sentencia (las reglas de
  `metadatos.py` y los dos jueces LLM); la referencia es la mayoría (2 de 3). Sin mayoría queda "no determinado" y la
  sentencia no cuenta para ese campo.
- **Relevancia de un par (consulta, sentencia)**, con la definición de la guía de anotación (2 = mismo delito y mismo
  problema procesal; 1 = solo uno de los dos; 0 = ninguno):
  * mismo delito: la familia de la categoría del STJ de la sentencia y la de la sentencia de la que salió la consulta
    (referencia externa; en la prueba piloto los jueces de 7B confundían "tentativa de homicidio" con "homicidio en
    grado de tentativa", así que su voto aquí solo se reporta);
  * mismo problema procesal: mayoría entre las reglas (mismo acto impugnado) y los dos jueces. La app ordena por
    familia y acto, así que las reglas favorecen a la app: por eso no deciden solas.
- **Consultas:** las comprobaciones deterministas de la guía (no copia la sentencia, no nombra el tipo del portal,
  sin datos identificables, no es una negativa del modelo) y los dos jueces deben aceptarlas.

Un juez se usa solo si acierta la familia de la categoría del STJ en al menos `config.JUEZ_EXACTITUD_MINIMA` de las
sentencias. Todo se escribe en `evaluacion/jueces/` y la evaluación lo lee en lugar de las hojas de anotación.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from .. import config
from ..metadatos import NO_DETERMINADO
from . import metricas as m
from .familias import etiqueta_familia

CAMPOS = ("acto_impugnado", "apelante", "sentido")
REGLAS, CONSENSO = "reglas", "consenso"
REFERENCIAS_PATH = config.JUECES_DIR / "referencias_sentencias.csv"
JUICIOS_AUTO_PATH = config.JUECES_DIR / "juicios_automaticos.csv"
CONSULTAS_VALIDADAS_PATH = config.JUECES_DIR / "consultas_validadas.csv"


def lector(modelo: str) -> str:
    return f"juez:{modelo}"


def mayoria(valores: list) -> str:
    """El valor que tiene mayoría absoluta entre TODOS los lectores (2 de 3, o 2 de 2); un lector sin respuesta
    válida (None) no vota pero sí cuenta. Sin mayoría: "no determinado"."""
    conteo = pd.Series([v for v in valores if v is not None], dtype=object).value_counts()
    if conteo.empty or conteo.iloc[0] < len(valores) // 2 + 1:
        return NO_DETERMINADO
    return conteo.index[0]


def particion_documento(doc_id: str) -> str:
    """Partición fija de cada sentencia para evaluar las fichas (mitad y mitad, por hash del id)."""
    return "prueba" if int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 2 else "desarrollo"


# --------------------------------------------------------------------------------------------------------------
# Sentencias
# --------------------------------------------------------------------------------------------------------------


def tabla_sentencias(penal: pd.DataFrame, familias: dict[str, str], metadatos: dict[str, dict],
                     respuestas: dict[str, dict[str, dict | None]]) -> pd.DataFrame:
    """Una fila por sentencia: familia del STJ y lo que leyó cada lector. `respuestas` = {modelo: {doc_id:
    respuesta normalizada o None}} (sentencias sin juzgar: ausentes)."""
    filas = []
    for d, tipo in zip(penal["id"], penal["tipo"]):
        meta = metadatos.get(d, {})
        f = {"doc_id": d, "particion": particion_documento(d), "familia_stj": etiqueta_familia(familias[tipo])}
        for campo in CAMPOS:
            f[f"{campo}:{REGLAS}"] = meta.get(campo, NO_DETERMINADO)
        for modelo, resp in respuestas.items():
            r = resp.get(d)
            f[f"familia:{lector(modelo)}"] = r["familia"] if r else None
            for campo in CAMPOS:
                f[f"{campo}:{lector(modelo)}"] = r[campo] if r else None
        filas.append(f)
    return pd.DataFrame(filas)


def validacion_jueces(tabla: pd.DataFrame, modelos: list[str],
                      minima: float = config.JUEZ_EXACTITUD_MINIMA) -> pd.DataFrame:
    """Exactitud de cada juez en la familia del delito contra la categoría del STJ (una respuesta inválida o
    ausente cuenta como error) y si alcanza el mínimo para usarse."""
    filas = []
    for modelo in modelos:
        col = f"familia:{lector(modelo)}"
        if col not in tabla:
            continue
        aciertos = (tabla[col] == tabla["familia_stj"]).astype(float).tolist()
        r = m.proporcion_con_ic(aciertos)
        filas.append({"juez": modelo, "sentencias": len(tabla), "con respuesta válida": int(tabla[col].notna().sum()),
                      "exactitud familia (STJ)": r["valor"], "IC95": f"[{r['ic_inf']:.3f}, {r['ic_sup']:.3f}]",
                      "mínimo": minima, "se usa": "sí" if r["valor"] >= minima else "no"})
    return pd.DataFrame(filas)


def usables(validacion: pd.DataFrame) -> list[str]:
    return validacion.loc[validacion["se usa"] == "sí", "juez"].tolist() if len(validacion) else []


def con_consenso(tabla: pd.DataFrame, jueces: list[str]) -> pd.DataFrame:
    t = tabla.copy()
    for campo in CAMPOS:
        cols = [f"{campo}:{REGLAS}"] + [f"{campo}:{lector(j)}" for j in jueces]
        t[f"{campo}:{CONSENSO}"] = [mayoria(list(v)) for v in t[cols].itertuples(index=False)]
    return t


def acuerdo_lectores(tabla: pd.DataFrame, jueces: list[str]) -> pd.DataFrame:
    """Kappa de Cohen y acuerdo simple entre cada par de lectores, por campo, sobre las sentencias donde ambos
    determinaron un valor (las reglas no leen el sentido en muchas sentencias: abstenerse no cuenta como desacuerdo)."""
    lectores = [REGLAS] + [lector(j) for j in jueces]
    filas = []
    for campo in CAMPOS:
        for a, b in itertools.combinations(lectores, 2):
            ca, cb = f"{campo}:{a}", f"{campo}:{b}"
            if ca not in tabla or cb not in tabla:
                continue
            t = tabla[[ca, cb]].dropna()
            t = t[(t[ca] != NO_DETERMINADO) & (t[cb] != NO_DETERMINADO)]
            if t.empty:
                continue
            filas.append({"campo": campo, "lector A": a, "lector B": b, "sentencias": len(t),
                          "acuerdo": float((t[ca] == t[cb]).mean()), "kappa": m.cohen_kappa(t[ca].tolist(),
                                                                                           t[cb].tolist())})
    return pd.DataFrame(filas)


def cobertura_consenso(tabla: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"campo": c, "sentencias": len(tabla),
                          "con referencia (mayoría)": int((tabla[f"{c}:{CONSENSO}"] != NO_DETERMINADO).sum())}
                         for c in CAMPOS])


# --------------------------------------------------------------------------------------------------------------
# Relevancia
# --------------------------------------------------------------------------------------------------------------


def _si_no(v: bool | None) -> str:
    return "" if v is None else "sí" if v else "no"


def juicios_automaticos(pares: pd.DataFrame, consultas: dict, doc_familia: dict[str, str], actos: dict[str, str],
                        respuestas: dict[str, dict[str, dict | None]], jueces: list[str]) -> pd.DataFrame:
    """Hoja de juicios con el formato de la del abogado (consulta_id, doc_id, relevancia, mismo_delito,
    mismo_problema_procesal, anotador): una fila por lector (todos los jueces con respuesta, para medir el acuerdo) y
    una del consenso con las reglas y los jueces usables `jueces` (solo si respondieron todos)."""
    filas = []
    for r in pares.itertuples():
        c = consultas[r.consulta_id]
        acto_c = actos.get(c.doc_id_fuente) if c.doc_id_fuente else None
        delito = doc_familia.get(r.doc_id) == c.familia
        proc = acto_c not in (None, "otro", NO_DETERMINADO) and actos.get(r.doc_id) == acto_c
        votos = [{"relevancia": 2 if delito and proc else 1 if delito or proc else 0, "mismo_delito": delito,
                  "mismo_problema_procesal": proc, "anotador": REGLAS}]
        for j in respuestas:
            resp = respuestas[j].get(f"{r.consulta_id}|{r.doc_id}")
            if resp:
                votos.append({**resp, "anotador": lector(j)})
        filas += [{"consulta_id": r.consulta_id, "doc_id": r.doc_id, "relevancia": v["relevancia"],
                   "mismo_delito": _si_no(v["mismo_delito"]),
                   "mismo_problema_procesal": _si_no(v["mismo_problema_procesal"]), "anotador": v["anotador"]}
                  for v in votos]
        validos = [v for v in votos if v["anotador"] in [REGLAS] + [lector(j) for j in jueces]]
        if len(validos) == 1 + len(jueces):
            p = mayoria([v["mismo_problema_procesal"] for v in validos])
            p = None if p == NO_DETERMINADO else bool(p)
            filas.append({"consulta_id": r.consulta_id, "doc_id": r.doc_id,
                          "relevancia": 2 if delito and p else 1 if delito or p else 0, "mismo_delito": _si_no(delito),
                          "mismo_problema_procesal": _si_no(p), "anotador": CONSENSO})
    return pd.DataFrame(filas, columns=["consulta_id", "doc_id", "relevancia", "mismo_delito",
                                        "mismo_problema_procesal", "anotador"])


def acuerdo_relevancia(juicios: pd.DataFrame) -> pd.DataFrame:
    """Kappa (grado 0/1/2 y relevante sí/no) entre cada par de lectores, sobre los pares que juzgaron ambos."""
    lectores = [a for a in juicios["anotador"].unique() if a != CONSENSO]
    filas = []
    for a, b in itertools.combinations(lectores, 2):
        pa = juicios[juicios["anotador"] == a].set_index(["consulta_id", "doc_id"])["relevancia"]
        pb = juicios[juicios["anotador"] == b].set_index(["consulta_id", "doc_id"])["relevancia"]
        comunes = pa.index.intersection(pb.index)
        if len(comunes):
            x, y = pa[comunes].astype(int).tolist(), pb[comunes].astype(int).tolist()
            filas.append({"lector A": a, "lector B": b, "pares": len(comunes),
                          "acuerdo exacto": float(np.mean(np.array(x) == np.array(y))),
                          "kappa 0/1/2": m.cohen_kappa(x, y),
                          "kappa relevante sí/no": m.cohen_kappa([v >= 1 for v in x], [v >= 1 for v in y])})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------------------------------------------
# Consultas
# --------------------------------------------------------------------------------------------------------------


def validar_consultas(consultas: list, textos: dict[str, str], respuestas: dict[str, dict[str, dict | None]],
                      jueces: list[str]) -> pd.DataFrame:
    """Una fila por consulta: validada (sí/no) y por qué no. Deterministas (las de la guía) + cada juez usable debe
    aceptarla (es una consulta; delito identificable, salvo en las consultas sin respuesta)."""
    from ..pipeline import es_fallo
    from .anotacion import _problemas

    filas = []
    for c in consultas:
        if not c.consulta.strip():
            continue
        if c.doc_id_fuente:
            motivo = _problemas(c, textos.get(c.doc_id_fuente, ""), [])
        else:
            motivo = "no era una consulta (el modelo se negó o respondió otra cosa)" if es_fallo(c.consulta) else ""
        for j in jueces:
            if motivo:
                break
            r = respuestas.get(j, {}).get(c.id)
            if r is None:
                motivo = f"{j}: sin respuesta válida"
            elif not r["es_consulta"]:
                motivo = f"{j}: no es una consulta de defensa coherente"
            elif not c.sin_respuesta and r["delito_identificable"] is False:
                motivo = f"{j}: no se identifica el delito"
        filas.append({"id": c.id, "particion": c.particion, "sin_respuesta": c.sin_respuesta,
                      "validada": not motivo, "motivo": motivo})
    return pd.DataFrame(filas, columns=["id", "particion", "sin_respuesta", "validada", "motivo"])


def leer_validadas(path=CONSULTAS_VALIDADAS_PATH) -> dict[str, bool] | None:
    """{id: validada} si ya se corrió `referencias`; None si no."""
    if not path.exists():
        return None
    t = pd.read_csv(path, dtype={"id": str})
    return dict(zip(t["id"], t["validada"].astype(bool)))


# --------------------------------------------------------------------------------------------------------------
# Fichas contra la referencia
# --------------------------------------------------------------------------------------------------------------


def comparar_fichas(fichas: pd.DataFrame, ref: pd.DataFrame, textos: dict[str, str] | None = None) -> pd.DataFrame:
    """Una fila por sentencia con ficha válida: lo que dice la ficha de la app (normalizado a los mismos valores
    cerrados) junto a la referencia, y si coinciden. Con `textos`, también los artículos del fundamento legal de la
    ficha frente a los que la sentencia invoca como fundamento (`fichas.fundamento_de_sentencia`)."""
    from ..fichas import FichaPenal, extraer_articulos, familia_del_delito, fundamento_de_sentencia
    from ..metadatos import _acto, _apelante

    r = ref.set_index("doc_id")
    filas = []
    for row in fichas[fichas["json_valido"].astype(bool)].itertuples():
        if row.doc_id not in r.index:
            continue
        f, x = FichaPenal.desde_guardada(row.ficha), r.loc[row.doc_id]
        pred = {"familia": etiqueta_familia(familia_del_delito(f.delitos)), "acto_impugnado": _acto(f.acto_impugnado),
                "apelante": _apelante(f.apelante), "sentido": f.sentido_fallo}
        fila = {"doc_id": row.doc_id, "particion": x["particion"], "familia_ref": x["familia_stj"],
                "familia_ficha": pred["familia"], "familia_ok": pred["familia"] == x["familia_stj"]}
        for c in CAMPOS:
            ref_c = x[f"{c}:{CONSENSO}"]
            fila.update({f"{c}_ref": ref_c, f"{c}_ficha": pred[c],
                         f"{c}_ok": None if ref_c == NO_DETERMINADO else pred[c] == ref_c})
        fila["articulos_no_encontrados"] = len(json.loads(row.articulos_no_encontrados))
        fila["modelo"] = getattr(row, "modelo", None)
        if textos is not None and row.doc_id in textos:
            citados = set().union(set(), *(extraer_articulos(x) for x in f.fundamento_legal))
            invocados = fundamento_de_sentencia(textos[row.doc_id])
            fila.update({"fundamento_ficha": len(citados), "fundamento_sentencia": len(invocados),
                         "fundamento_comunes": len(citados & invocados)})
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_fundamento(comp: pd.DataFrame) -> pd.DataFrame:
    """Precisión y exhaustividad (recall) de los artículos que cita la ficha frente a los que la sentencia invoca como
    fundamento, por ficha y promediadas, con IC 95 % por bootstrap. Solo fichas que citan algún artículo y sentencias
    en que las reglas leen fundamento."""
    if "fundamento_ficha" not in comp:
        return pd.DataFrame()
    c = comp[(comp["fundamento_ficha"] > 0) & (comp["fundamento_sentencia"] > 0)]
    precision = (c["fundamento_comunes"] / c["fundamento_ficha"]).tolist()
    recall = (c["fundamento_comunes"] / c["fundamento_sentencia"]).tolist()
    filas = []
    for nombre, valores, lectura in (
            ("precisión", precision, "de los artículos que cita la ficha, fracción que la Sala invoca como fundamento"),
            ("exhaustividad (recall)", recall, "de los artículos que invoca la Sala, fracción que recoge la ficha; baja "
                                               "por construcción: la ficha resume los principales"),
            ("fichas con todas sus citas en el fundamento", [float(p == 1.0) for p in precision], "")):
        inf, sup = m.bootstrap_ic(valores)
        filas.append({"medida": nombre, "fichas": len(valores),
                      "valor": float(np.mean(valores)) if valores else float("nan"),
                      "IC95": f"[{inf:.3f}, {sup:.3f}]" if valores else "—", "lectura": lectura})
    filas.append({"medida": "fichas sin artículos citados", "fichas": int((comp.get("fundamento_ficha", 0) == 0).sum()),
                  "valor": float("nan"), "IC95": "—", "lectura": "la ficha no cita fundamento legal"})
    return pd.DataFrame(filas)


def tabla_exactitud_fichas(comp: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c, ref in (("familia", "categoría del STJ"), ("acto_impugnado", "mayoría de lectores"),
                   ("apelante", "mayoría de lectores"), ("sentido", "mayoría de lectores")):
        ok = comp[f"{c}_ok"].dropna().astype(float).tolist()
        r = m.proporcion_con_ic(ok) if ok else {"valor": float("nan"), "ic_inf": float("nan"), "ic_sup": float("nan")}
        filas.append({"campo": c, "referencia": ref, "sentencias con referencia": len(ok),
                      "exactitud": r["valor"], "IC95": f"[{r['ic_inf']:.3f}, {r['ic_sup']:.3f}]" if ok else "—"})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------------------------------------------


def construir(modelos: list[str] | None = None, log=print, completa: bool = False):
    """Junta reglas y jueces en las tres referencias (sentencias, juicios, consultas validadas) y deja constancia en
    una corrida "referencias": validación de cada juez, acuerdo entre lectores y cobertura. Sin LLM."""
    from .. import metadatos as md
    from ..datos.corpus import load_corpus
    from . import jueces as J
    from .conjuntos import cargar_todas
    from .familias import cargar_familias
    from .reporte import Corrida

    modelos = modelos or config.MODELOS_JUECES
    corpus = load_corpus()
    penal = corpus[corpus["materia"] == "penal"]
    familias = cargar_familias(corpus["tipo"].unique().tolist())
    textos = corpus.set_index("id")["texto"].to_dict()
    consultas = cargar_todas(corpus=corpus, familias=familias, permitir_vacias=True)
    meta = md.cargar()
    if not meta:
        raise SystemExit("❌ Faltan los metadatos procesales: corre `copiloto-datos metadatos`.")
    pares = pd.read_csv(J.PARES_PATH, dtype=str) if J.PARES_PATH.exists() else pd.DataFrame(
        columns=["consulta_id", "consulta", "doc_id"])
    # El prompt del juez lleva la lista de familias de TODO el corpus penal: los elementos se arman con todo y luego
    # se filtra la muestra (armarlos con la muestra cambiaría el prompt y ninguna respuesta guardada coincidiría).
    items_sent = J.items_sentencias(penal, familias)
    items_todas, penal_todo = items_sent, penal
    if not completa:  # misma muestra que los jueces: la referencia solo existe donde todos juzgaron
        items_sent = [it for it in items_sent if J.de_la_muestra(it)]
        penal = penal[[J.en_muestra(d, config.MUESTRA_SENTENCIAS) for d in penal["id"]]]
        consultas = [c for c in consultas if J.en_muestra(c.id, config.MUESTRA_CONSULTAS)]
        pares = pares[[J.en_muestra(c, config.MUESTRA_CONSULTAS) for c in pares["consulta_id"]]]
    items = {"sentencia": items_sent, "relevancia": J.items_relevancia(pares, textos, corpus.set_index("id")["tipo"].to_dict()),
             "consulta": J.items_consultas(consultas)}
    corrida = Corrida("referencias", parametros={
        "jueces": modelos, "exactitud_minima": config.JUEZ_EXACTITUD_MINIMA, "muestra": "completa" if completa else {
            "sentencias": config.MUESTRA_SENTENCIAS, "consultas": config.MUESTRA_CONSULTAS, "semilla": config.EVAL_SEED,
            "n_sentencias": len(penal), "n_consultas": len(consultas), "n_pares": len(pares)}})
    respuestas: dict[str, dict[str, dict]] = {}
    completas: dict[str, dict] = {}  # jueces que ya juzgaron TODAS las sentencias: se validan con todas
    for modelo in modelos:
        registros = J.leer(modelo)
        if not registros:
            corrida.advertir(f"{modelo}: sin respuestas guardadas (corre `jueces --modelo {modelo}`).")
            continue
        respuestas[modelo] = {}
        for tarea, its in items.items():
            v = J.vigentes(its, registros)
            respuestas[modelo][tarea] = {i: r for (_, i), r in v.items()}
            if len(v) < len(its):
                corrida.advertir(f"{modelo}: {len(its) - len(v)} de {len(its)} elementos de '{tarea}' sin juzgar con "
                                 "el prompt y el texto actuales.")
        if not completa:
            v_todas = J.vigentes(items_todas, registros)
            if len(v_todas) == len(items_todas):
                completas[modelo] = {i: r for (_, i), r in v_todas.items()}
        corrida.extra[f"digest_{modelo}"] = next((r.get("digest") for r in registros.values()), None)

    tabla = tabla_sentencias(penal, familias, meta, {j: r["sentencia"] for j, r in respuestas.items()})
    # El mínimo se fijó sobre todas las sentencias: si el juez las juzgó todas, se valida con todas (más precisa que la
    # muestra y sin elegir la cifra favorable); si no, con la muestra.
    val = pd.concat([validacion_jueces(tabla_sentencias(penal_todo, familias, meta, {j: completas[j]})
                                       if j in completas else tabla, [j]) for j in respuestas] or [pd.DataFrame()],
                    ignore_index=True)
    jueces = usables(val)
    if not jueces:
        corrida.advertir("Ningún juez usable: la referencia queda solo con las reglas (no es una evaluación "
                         "independiente).")
    tabla = con_consenso(tabla, jueces)
    config.JUECES_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REFERENCIAS_PATH, index=False)
    corrida.tabla("Validación de los jueces", "Exactitud de la familia del delito que lee cada juez contra la "
                  "categoría que asignó el STJ (referencia externa; una respuesta inválida cuenta como error). Un juez "
                  "se usa si alcanza el mínimo, fijado antes de correrlos.", val)
    corrida.tabla("Acuerdo entre lectores de la sentencia", "Reglas y jueces leen acto impugnado, apelante y "
                  "sentido; kappa de Cohen sobre las sentencias donde ambos respondieron.",
                  acuerdo_lectores(tabla, list(respuestas)))
    corrida.tabla("Cobertura de la referencia", "Sentencias con mayoría absoluta entre reglas y jueces usables "
                  f"({', '.join(jueces) or 'ninguno'}); sin mayoría no cuentan.", cobertura_consenso(tabla))

    doc_familia = {d: familias[t] for d, t in zip(penal["id"], penal["tipo"])}
    actos = {d: x.get("acto_impugnado", NO_DETERMINADO) for d, x in meta.items()}
    por_id = {c.id: c for c in consultas}
    juicios = juicios_automaticos(pares[pares["consulta_id"].isin(por_id)], por_id, doc_familia, actos,
                                  {j: r["relevancia"] for j, r in respuestas.items()}, jueces)
    juicios.to_csv(JUICIOS_AUTO_PATH, index=False)
    if len(juicios):
        corrida.tabla("Acuerdo en la relevancia", "Entre cada par de lectores, sobre los mismos pares (consulta, "
                      "sentencia). La referencia sale de las facetas: mismo delito por la categoría del STJ y mismo problema "
                      "procesal por mayoría.", acuerdo_relevancia(juicios))
        cons = juicios[juicios["anotador"] == CONSENSO]
        corrida.tabla("Pares juzgados", "Pares (consulta, sentencia) por agrupamiento: los 10 primeros de la app y los 5 "
                      "primeros de BM25 y de embeddings.", pd.DataFrame([{
                          "pares": len(pares), "con referencia": len(cons),
                          **{f"relevancia {g}": int((cons["relevancia"] == g).sum()) for g in (0, 1, 2)}}]))
    else:
        corrida.advertir("Sin pares juzgados: corre `jueces --pares` y los jueces.")

    val_c = validar_consultas(consultas, textos, {j: r["consulta"] for j, r in respuestas.items()}, jueces)
    val_c.to_csv(CONSULTAS_VALIDADAS_PATH, index=False)
    resumen = val_c.groupby(["particion", "sin_respuesta"]).agg(consultas=("id", "size"),
                                                                 validadas=("validada", "sum")).reset_index()
    corrida.tabla("Consultas validadas", "Sustituye a la revisión humana: comprobaciones de la guía y aceptación de "
                  "cada juez usable. Solo las validadas entran a la evaluación.", resumen)
    motivos = val_c[~val_c["validada"]]
    if len(motivos):
        corrida.tabla("Consultas descartadas", "Por qué no pasaron la validación.", motivos, diagnostico=True)
    log(f"Referencias en {config.JUECES_DIR}")
    return corrida


def evaluar_fichas(particion: str, log=print):
    """Fichas precalculadas de la app contra la referencia: familia del delito (STJ), acto impugnado, apelante y
    sentido (mayoría de lectores), y artículos citados que no están en la sentencia."""
    from ..fichas_precalculadas import cargar
    from .reporte import Corrida

    if not REFERENCIAS_PATH.exists():
        raise SystemExit("❌ Faltan las referencias: corre `referencias`.")
    ref = pd.read_csv(REFERENCIAS_PATH, dtype=str).fillna(NO_DETERMINADO)
    if particion != "todas":
        ref = ref[ref["particion"] == particion]
    from ..datos.corpus import load_corpus

    corpus = load_corpus()
    comp = comparar_fichas(cargar(), ref, corpus.set_index("id")["texto"].to_dict())
    corrida = Corrida("fichas-referencia", parametros={"particion": particion})
    corrida.tabla("Exactitud de las fichas", "Ficha precalculada (la que muestra la app) contra la referencia "
                  "automática; IC95 de Wilson. Solo cuentan las sentencias con referencia.", tabla_exactitud_fichas(comp))
    corrida.tabla("Fundamento legal de la ficha", "Artículos que cita la ficha frente a los que la sentencia invoca "
                  "como fundamento («con fundamento en», «de conformidad con», «en términos de», «con base en» y puntos "
                  "resolutivos), leídos por reglas. Promedio por ficha; IC95 por bootstrap. No distingue el "
                  "ordenamiento (Código Penal o Código Nacional de Procedimientos Penales).", tabla_fundamento(comp))
    faltan = comp["articulos_no_encontrados"] if len(comp) else pd.Series(dtype=int)
    corrida.tabla("Artículos que no aparecen en la sentencia", "Posibles alucinaciones (o artículos en páginas que "
                  "el PDF no trae).", pd.DataFrame([{"fichas": len(comp), "fichas con alguno": int((faltan > 0).sum()),
                                                    "artículos no encontrados": int(faltan.sum())}]))
    errores = comp[(comp["sentido_ok"] == False) | (comp["familia_ok"] == False)]  # noqa: E712
    if len(errores):
        corrida.tabla("Fichas que no coinciden", "Sentido o familia distintos de la referencia.",
                      errores[["doc_id", "familia_ref", "familia_ficha", "sentido_ref", "sentido_ficha"]],
                      diagnostico=True)
    log(f"{len(comp)} fichas comparadas ({particion}).")
    return corrida
