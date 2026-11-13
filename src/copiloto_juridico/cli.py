"""Línea de comandos para preparar los datos.

    python -m copiloto_juridico.cli construir-corpus --chroma ../chroma_db_juridica_8.2 --pdfs ruta/a/pdfs
    python -m copiloto_juridico.cli indexar            # embeddings + metadatos procesales por reglas
    python -m copiloto_juridico.cli descargar          # descarga balanceada (si la API lo permite)
    python -m copiloto_juridico.cli estado
    python -m copiloto_juridico.cli pendientes --materia penal   # qué descargar a mano, por prioridad
"""

from __future__ import annotations

import argparse
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
    from .datos.corpus import import_from_chroma_sqlite, load_corpus, save_corpus
    from .datos.procesamiento import process_pdf_folder

    fuentes = []
    if not args.reiniciar:
        actual = load_corpus()
        if len(actual):
            _log(f"Corpus actual: {len(actual)} sentencias.")
            fuentes.append(actual)
    for db in args.chroma or []:
        df = import_from_chroma_sqlite(Path(db))
        _log(f"Importadas {len(df)} sentencias desde ChromaDB '{db}'.")
        fuentes.append(df)
    for carpeta in args.pdfs or []:
        df = pd.DataFrame(process_pdf_folder(Path(carpeta), log=_log))
        _log(f"Procesadas {len(df)} sentencias desde PDFs en '{carpeta}'.")
        fuentes.append(df)
    if not fuentes:
        sys.exit("No hay fuentes: usa --chroma y/o --pdfs.")

    todo = pd.concat(fuentes, ignore_index=True)
    # Si una sentencia viene de varias fuentes se conserva la primera (corpus actual > chroma > PDFs).
    todo = todo.drop_duplicates(subset="id", keep="first")
    todo = todo[todo["texto"].str.len() >= 500]
    balanceado = select_balanced(todo, args.max_por_tipo)
    save_corpus(balanceado)

    rep = balance_report(balanceado, load_catalogo())
    _log(f"\nCorpus balanceado: {len(balanceado)} sentencias de {len(todo)} disponibles, "
         f"{balanceado['tipo'].nunique()} tipos (tope {args.max_por_tipo} por tipo).")
    _log(balanceado.groupby("materia").size().rename("sentencias").to_string())
    _log(f"Tipos del catálogo completos: {int(rep['completo'].sum())}/{len(rep)}")


def cmd_indexar(args) -> None:
    from .datos.corpus import load_corpus
    from .datos.indice import build_index

    corpus = load_corpus()
    if corpus.empty:
        sys.exit("El corpus está vacío. Ejecuta primero 'construir-corpus'.")
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


def cmd_descargar(args) -> None:
    from .datos.corpus import load_corpus
    from .datos.descarga import VerificacionRequerida, descargar_balanceado

    corpus = load_corpus()
    try:
        descargar_balanceado(Path(args.destino), log=_log, ya_en_corpus=Counter(corpus["tipo"]))
    except VerificacionRequerida as e:
        sys.exit(f"❌ {e}")


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
    """Aplica la limpieza v11 al corpus guardado (sin necesidad de los PDFs) y lo vuelve a guardar."""
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
    _log("\nLa API exige reCAPTCHA: descarga los PDFs desde el portal, cópialos a la carpeta de PDFs y corre "
         "`construir-corpus --pdfs <carpeta>` y luego `indexar`. Congela el corpus antes de medir.")


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

    sub.add_parser("indexar", help="Genera los embeddings del corpus en data/indice (y los metadatos procesales)"
                   ).set_defaults(func=cmd_indexar)
    sub.add_parser("metadatos", help="Solo los metadatos procesales por reglas y su cobertura"
                   ).set_defaults(func=cmd_metadatos)

    d = sub.add_parser("descargar", help="Descarga balanceada desde la API del STJ Jalisco")
    d.add_argument("--destino", default=str(config.RAW_PDF_DIR))
    d.set_defaults(func=cmd_descargar)

    es = sub.add_parser("estado", help="Muestra el balance del corpus")
    es.add_argument("--salida", help="CSV con el balance por tipo")
    es.set_defaults(func=cmd_estado)

    lc = sub.add_parser("limpiar-corpus", help="Limpieza v11 del texto guardado (anexo testado, etiquetas, encabezados)")
    lc.add_argument("--omitidas", help="CSV con las sentencias que tienen páginas omitidas")
    lc.set_defaults(func=cmd_limpiar_corpus)

    pe = sub.add_parser("pendientes", help="Tipos incompletos por prioridad (para descargar a mano)")
    pe.add_argument("--materia", default="penal")
    pe.add_argument("--top", type=int, default=30)
    pe.add_argument("--salida", help="CSV con la lista completa")
    pe.set_defaults(func=cmd_pendientes)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
