"""Línea de comandos para preparar los datos.

    python -m copiloto_juridico.cli construir-corpus --chroma ../chroma_db_juridica_8.2 --pdfs ruta/a/pdfs
    python -m copiloto_juridico.cli indexar            # embeddings + metadatos procesales por reglas
    python -m copiloto_juridico.cli estado
    python -m copiloto_juridico.cli pendientes --materia penal   # qué descargar a mano, por prioridad
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_construir_corpus(args) -> None:
    from .datos.balance import balance_report, load_catalogo, select_balanced
    from .datos.corpus import (descargas_ajenas, duplicados_de_texto, import_from_chroma_sqlite, load_corpus,
                               save_corpus, unir_fuentes)
    from .datos.procesamiento import process_pdf_folder

    # Orden de las fuentes: PDFs > corpus actual > ChromaDB. El PDF procesado ahora reemplaza al texto guardado de la
    # misma sentencia (así entra la extracción corregida); los PDF con el texto de otra sentencia ya se quitaron.
    fuentes, viejas, actual, rechazadas = [], [], pd.DataFrame(columns=["id"]), []
    if not args.reiniciar:
        actual = load_corpus()
        if len(actual):
            _log(f"Corpus actual: {len(actual)} sentencias.")
            viejas.append(actual)
    for db in args.chroma or []:
        df = import_from_chroma_sqlite(Path(db))
        _log(f"Importadas {len(df)} sentencias desde ChromaDB '{db}'.")
        viejas.append(df)
    for carpeta in args.pdfs or []:
        df = pd.DataFrame(process_pdf_folder(Path(carpeta), log=_log))
        _log(f"Procesadas {len(df)} sentencias desde PDFs en '{carpeta}'.")
        if len(actual) and len(df):
            ajenas = descargas_ajenas(actual, df)
            if len(ajenas):
                _log(f"⚠️ {len(ajenas)} PDF con el nombre de una sentencia del corpus pero el texto de otra; se conserva "
                     f"el texto anterior: {', '.join(ajenas['id'])}")
                rechazadas.append(ajenas.assign(carpeta=str(carpeta)))
                df = df[~df["id"].isin(ajenas["id"])]
        fuentes.append(df)
    fuentes += viejas
    if not fuentes:
        sys.exit("No hay fuentes: usa --chroma y/o --pdfs.")
    if args.pdfs:  # el reporte de esta construcción, de todas las carpetas (vacío si no hubo rechazos)
        (pd.concat(rechazadas) if rechazadas else pd.DataFrame(columns=["id", "carpeta"])).to_csv(
            config.CORPUS_DIR / "descargas_rechazadas.csv", index=False)

    # Una sentencia repetida: gana la versión sin páginas omitidas (el PDF vuelto a descargar reemplaza a la
    # muestreada); entre iguales, la primera fuente (PDFs > corpus actual > chroma).
    todo = unir_fuentes(fuentes)
    todo = todo[todo["texto"].str.len() >= 500]
    # El corpus contiene solo las materias del alcance (penal); las demás ya no se guardan ni se indexan.
    fuera = ~todo["materia"].isin(config.MATERIAS_ACTIVAS)
    if fuera.any():
        _log(f"Fuera del alcance ({', '.join(config.MATERIAS_ACTIVAS)}): {int(fuera.sum())} sentencias de "
             f"{', '.join(sorted(todo.loc[fuera, 'materia'].unique()))}.")
        todo = todo[~fuera]
    # La misma sentencia publicada dos veces con distinto nombre: contaría doble en la búsqueda y podía quedar una
    # copia en desarrollo y otra en prueba. Se queda la que es fuente de consultas o, si no, la primera por fecha.
    fuentes_consultas = set()
    if config.CONSULTAS_PATH.exists():
        fuentes_consultas = {json.loads(x).get("doc_id_fuente") for x in config.CONSULTAS_PATH.read_text(
            encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")}
    dup = duplicados_de_texto(todo, fuentes_consultas)
    if args.pdfs or args.chroma:  # el de esta construcción con fuentes nuevas, aunque esté vacío
        dup.to_csv(config.CORPUS_DIR / "duplicados.csv", index=False)
    if len(dup):
        _log(f"⚠️ {len(dup)} sentencias repetidas (mismo texto con otro nombre); se quita la copia.")
        todo = todo[~todo["id"].isin(dup["id"])]
    huerfanas = sorted(set(dup["id"]) & fuentes_consultas)
    if huerfanas:
        _log(f"⚠️ Consultas con fuente repetida: apúntalas a la copia que se queda (duplicados.csv): {huerfanas}")
    balanceado = select_balanced(todo, args.max_por_tipo, conservar=set(actual["id"]))
    save_corpus(balanceado)

    rep = balance_report(balanceado, load_catalogo())
    _log(f"\nCorpus balanceado: {len(balanceado)} sentencias de {len(todo)} disponibles, "
         f"{balanceado['tipo'].nunique()} tipos (tope {args.max_por_tipo} por tipo).")
    _log(balanceado.groupby("materia").size().rename("sentencias").to_string())
    _log(f"Tipos del catálogo completos: {int(rep['completo'].sum())}/{len(rep)}")


def cmd_indexar(args) -> None:
    from .datos.corpus import load_corpus
    from .datos.indice import build_index, recortar_indice

    corpus = load_corpus()
    if corpus.empty:
        sys.exit("El corpus está vacío. Ejecuta primero 'construir-corpus'.")
    if args.recortar:
        try:
            recortar_indice(corpus, log=_log)
        except ValueError as e:
            sys.exit(f"❌ {e}")
    else:
        build_index(corpus, log=_log)
    cmd_metadatos(args, corpus)  # los metadatos procesales van siempre con el corpus indexado


def cmd_metadatos(args, corpus=None) -> None:
    """Acto impugnado, apelante y sentido por reglas para el corpus penal, y su cobertura (diagnóstico)."""
    from . import metadatos
    from .datos.corpus import load_corpus
    from .evaluacion.reporte import Corrida

    tabla = metadatos.calcular(load_corpus() if corpus is None else corpus)
    path = metadatos.guardar(tabla)
    corrida = Corrida("metadatos-cobertura", parametros={"sentencias_penales": len(tabla)})
    corrida.tabla("Cobertura de los metadatos procesales (reglas)",
                  "Valores leídos por reglas del párrafo inicial (acto impugnado, quién apeló) y de los puntos "
                  "resolutivos (sentido). 'no determinado' = las reglas no lo encontraron. Sin anotación: no mide "
                  "exactitud.", metadatos.cobertura(tabla), diagnostico=True)
    pj, _ = corrida.guardar()
    for campo in ("acto_impugnado", "apelante", "sentido"):
        nd = int((tabla[campo] == metadatos.NO_DETERMINADO).sum())
        _log(f"   {campo}: {len(tabla) - nd}/{len(tabla)} determinados ({nd} no determinados)")
    _log(f"Metadatos procesales: {path} · cobertura en {pj.name}")


def cmd_estado(args) -> None:
    from .datos.balance import balance_report, load_catalogo
    from .datos.corpus import load_corpus

    corpus = load_corpus()
    _log(f"Sentencias en el corpus: {len(corpus)} | Tipos: {corpus['tipo'].nunique()}")
    _log(corpus.groupby("materia").size().to_string())
    rep = balance_report(corpus, load_catalogo())
    _log(f"Tipos del catálogo completos: {int(rep['completo'].sum())}/{len(rep)} | "
         f"Sentencias faltantes para el balance total: {int(rep['faltan'].sum())}")
    if args.salida:
        rep.to_csv(args.salida, index=False)
        _log(f"Reporte de balance por tipo: {args.salida}")


def cmd_limpiar_corpus(args) -> None:
    """Aplica la limpieza al corpus guardado (sin necesidad de los PDFs) y lo vuelve a guardar."""
    from .datos.corpus import load_corpus, save_corpus
    from .datos.procesamiento import limpiar_texto

    corpus = load_corpus()
    antes = corpus["texto"].str.len()
    corpus["texto"] = corpus["texto"].map(limpiar_texto)
    despues = corpus["texto"].str.len()
    save_corpus(corpus)
    _log(f"Corpus limpio: {len(corpus)} sentencias, {antes.sum():,} -> {despues.sum():,} caracteres "
         f"({1 - despues.sum() / antes.sum():.0%} de ruido quitado). Sigue con `indexar`.")
    omitidas = corpus[corpus["texto"].str.contains("págs? omitidas?|pág\\(s\\) omitida", regex=True)]
    if args.omitidas:
        omitidas[["id", "materia", "tipo"]].to_csv(args.omitidas, index=False)
        _log(f"{len(omitidas)} sentencias con páginas omitidas en la ingesta original: {args.omitidas} "
             "(vuelve a descargar esos PDF y procésalos completos con `construir-corpus --pdfs`).")


def cmd_redescarga(args) -> None:
    """Sentencias penales con páginas omitidas en la ingesta original, para volver a bajarlas a mano (la API pide
    reCAPTCHA). Orden: primero las familias con menos sentencias (cada una pesa más en la evaluación)."""
    import re

    from .datos.corpus import load_corpus
    from .evaluacion.familias import cargar_familias, etiqueta_familia
    from .metadatos import toca_de

    corpus = load_corpus()
    penal = corpus[corpus["materia"] == "penal"].copy()
    familias = cargar_familias(penal["tipo"].unique().tolist())
    penal["familia"] = penal["tipo"].map(familias)
    por_familia = penal["familia"].value_counts()
    faltan = penal[penal["texto"].str.contains("págs? omitidas?", regex=True)].copy()

    filas = pd.DataFrame({
        "id": faltan["id"], "fecha": faltan["fecha"], "tipo": faltan["tipo"],
        "familia": faltan["familia"].map(etiqueta_familia), "sentencias_familia": faltan["familia"].map(por_familia),
        "toca": faltan["texto"].map(toca_de),
        "paginas_omitidas": faltan["texto"].map(lambda t: sum(map(int, re.findall(r"(\d+) págs? omitidas?", t)))),
        "enlace": ""})  # el corpus vino de una base anterior sin el id de la API: se busca por toca y fecha
    filas = filas.sort_values(["sentencias_familia", "familia", "fecha", "id"])
    filas.to_csv(args.salida, index=False)
    _log(f"{len(filas)} sentencias penales por volver a descargar: {args.salida}\n"
         "Guarda cada PDF con el nombre de la columna id en una carpeta y corre:\n"
         "  copiloto-datos construir-corpus --pdfs CARPETA   (la versión completa reemplaza a la muestreada)\n"
         "  copiloto-datos indexar                          (embeddings + metadatos procesales)\n"
         "  python -m copiloto_juridico.evaluacion fichas-batch   (regenera solo las fichas cuyo texto cambió)\n"
         "  python -m copiloto_juridico.evaluacion congelar       (lo congelado ya no coincide)")


def cmd_pendientes(args) -> None:
    """Tipos que aún no llegan a su meta, ordenados por prioridad para la descarga manual.

    Prioridad: primero los tipos de familias con al menos 10 sentencias (donde la evaluación tendrá consultas)
    y, entre ellos, los que más sentencias pueden ganar. Los tipos con menos de 5 sentencias publicadas no pueden
    balancearse descargando: para ellos se evalúa a nivel familia.
    """
    from .datos.balance import balance_report, load_catalogo
    from .datos.corpus import load_corpus
    from .evaluacion.familias import cargar_familias

    corpus = load_corpus()
    rep = balance_report(corpus, load_catalogo())
    if args.materia:
        rep = rep[rep["materia"] == args.materia]
    rep = rep[rep["faltan"] > 0].copy()
    familias = cargar_familias(rep["tipo"].tolist())
    rep["familia"] = rep["tipo"].map(familias)
    en_corpus_fam = corpus.assign(familia=corpus["tipo"].map(cargar_familias(corpus["tipo"].unique().tolist())))
    tam = en_corpus_fam.groupby("familia").size()
    rep["sentencias_familia"] = rep["familia"].map(tam).fillna(0).astype(int)
    rep["familia_grande"] = rep["sentencias_familia"] >= 10
    rep = rep.sort_values(["familia_grande", "faltan", "sentencias_familia"], ascending=False)
    cols = ["tipo", "familia", "disponibles", "en_corpus", "meta", "faltan"]
    _log(f"{len(rep)} tipos incompletos · faltan {int(rep['faltan'].sum()):,} sentencias "
         f"(tope {config.MAX_SENTENCIAS_POR_TIPO} por tipo).")
    _log(rep[cols].head(args.top).to_string(index=False))
    if args.salida:
        rep[cols + ["sentencias_familia"]].to_csv(args.salida, index=False)
        _log(f"\nLista completa: {args.salida}")
    _log("\nDescarga los PDF con `python descarga_datos/descargar.py --modo navegador` (una persona resuelve el "
         "reCAPTCHA), corre `construir-corpus --pdfs <carpeta>` y luego `indexar`. Congela el corpus antes de medir.")


def cmd_demo(args) -> None:
    """Precalcula los ejemplos penales de la app para que la demostración no espere al LLM, y guarda el tiempo de
    cada uno sin caché (análisis completo) y con caché (lectura del resultado guardado)."""
    import time

    from . import cache_resultados
    from .clasificador import construir_clasificador
    from .ejemplos import EJEMPLOS
    from .evaluacion import metricas as m
    from .evaluacion.reporte import Corrida
    from .llm import get_client
    from .pipeline import Copiloto
    from .vectorstore import SentenciasStore

    if not config.CACHE_RESULTADOS:
        sys.exit("❌ CACHE_RESULTADOS está apagado (COPILOTO_CACHE_RESULTADOS=false): no hay dónde guardar la demo.")
    settings = config.LLMSettings()
    cliente = get_client(settings)
    ok, msg = cliente.health()
    if not ok:
        sys.exit(f"❌ {msg}")
    store = SentenciasStore(log=lambda _m: None)
    clas = construir_clasificador(store.corpus["tipo"].unique().tolist())
    filas = []
    for i, e in enumerate(EJEMPLOS[:args.max], 1):
        previo = cache_resultados.buscar(e.texto, settings)
        t0 = time.time()
        if previo is None or args.rehacer:
            logs, resultado = [], None
            for ev in Copiloto(store, clas, cliente).analizar(e.texto):
                if ev.tipo == "log":
                    logs.append(ev.mensaje)
                elif ev.tipo == "resultado":
                    resultado = ev.datos["resultado"]
            sin = time.time() - t0
            if resultado is not None:
                cache_resultados.guardar(resultado, logs, settings)
        else:
            sin = previo["resultado"].segundos  # ya estaba guardado: el tiempo del análisis original
        t1 = time.time()
        con = cache_resultados.buscar(e.texto, settings)
        filas.append({"ejemplo": e.etiqueta, "sin caché (s)": round(sin, 1),
                      "con caché (s)": round(time.time() - t1, 3) if con else None,
                      "guardado": "sí" if con else "no (sin resultado)"})
        _log(f"[{i}] {e.etiqueta}: {sin:.0f} s sin caché · {filas[-1]['con caché (s)']} s con caché")
    df = pd.DataFrame(filas)
    corrida = Corrida("demo", parametros={"ejemplos": len(filas), "rehacer": args.rehacer})
    corrida.tabla("Tiempo de la demo", "Por ejemplo penal de la app. Sin caché = análisis completo con el LLM (o el "
                  "tiempo del análisis original si ya estaba guardado); con caché = leer el resultado guardado, que "
                  "es lo que espera la app en la demo (sin contar el dibujado de Streamlit).", df, diagnostico=True)
    validos = df.dropna(subset=["con caché (s)"])
    corrida.tabla("Resumen", "p50 / p95 en segundos.", pd.DataFrame([
        {"modo": "sin caché", **m.percentiles(validos["sin caché (s)"].tolist())},
        {"modo": "con caché", **m.percentiles(validos["con caché (s)"].tolist())}]), diagnostico=True)
    pj, pm = corrida.guardar()
    _log(corrida.markdown() + f"\nGuardado en {pj}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="copiloto_juridico", description="Preparación de datos del Co-piloto Jurídico")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("construir-corpus", help="Une fuentes, aplica el balance y guarda el corpus en data/corpus")
    c.add_argument("--chroma", nargs="*", help="Rutas a ChromaDB antiguas (se lee su SQLite)")
    c.add_argument("--pdfs", nargs="*", help="Carpetas con PDFs nombrados como fecha_materia_delito.pdf")
    c.add_argument("--max-por-tipo", type=int, default=config.MAX_SENTENCIAS_POR_TIPO)
    c.add_argument("--reiniciar", action="store_true", help="Ignora el corpus actual")
    c.set_defaults(func=cmd_construir_corpus)

    ix = sub.add_parser("indexar", help="Genera los embeddings del corpus en data/indice (y los metadatos procesales)")
    ix.add_argument("--recortar", action="store_true",
                    help="Sin recalcular: conserva del índice actual solo los fragmentos de las sentencias del corpus "
                         "(cuando el corpus es un subconjunto con los mismos textos)")
    ix.set_defaults(func=cmd_indexar)
    sub.add_parser("metadatos", help="Solo los metadatos procesales por reglas y su cobertura"
                   ).set_defaults(func=cmd_metadatos)


    es = sub.add_parser("estado", help="Muestra el balance del corpus")
    es.add_argument("--salida", help="CSV con el balance por tipo")
    es.set_defaults(func=cmd_estado)

    lc = sub.add_parser("limpiar-corpus", help="Limpieza del texto guardado (anexo testado, etiquetas, encabezados)")
    lc.add_argument("--omitidas", help="CSV con las sentencias que tienen páginas omitidas")
    lc.set_defaults(func=cmd_limpiar_corpus)

    rd = sub.add_parser("redescarga", help="Sentencias penales con páginas omitidas, para volver a descargarlas")
    rd.add_argument("--salida", default=str(config.EVAL_DIR / "redescarga_penal.csv"))
    rd.set_defaults(func=cmd_redescarga)

    pe = sub.add_parser("pendientes", help="Tipos incompletos por prioridad (para descargar a mano)")
    pe.add_argument("--materia", default="penal")
    pe.add_argument("--top", type=int, default=30)
    pe.add_argument("--salida", help="CSV con la lista completa")
    pe.set_defaults(func=cmd_pendientes)

    dm = sub.add_parser("demo", help="Precalcula los ejemplos penales para la demostración (caché de resultados)")
    dm.add_argument("--ejemplos", action="store_true", required=True, help="Los ejemplos penales de la app")
    dm.add_argument("--max", type=int, default=None, help="Solo los primeros N ejemplos")
    dm.add_argument("--rehacer", action="store_true", help="Vuelve a analizar aunque ya estén guardados")
    dm.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
