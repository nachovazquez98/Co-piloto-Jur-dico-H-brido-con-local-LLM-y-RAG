"""Línea de comandos para preparar los datos.

    python -m copiloto_juridico.cli construir-corpus --chroma ../chroma_db_juridica_8.2 --pdfs ruta/a/pdfs
    python -m copiloto_juridico.cli indexar
    python -m copiloto_juridico.cli descargar          # descarga balanceada (si la API lo permite)
    python -m copiloto_juridico.cli estado
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

    sub.add_parser("indexar", help="Genera los embeddings del corpus en data/indice").set_defaults(func=cmd_indexar)

    d = sub.add_parser("descargar", help="Descarga balanceada desde la API del STJ Jalisco")
    d.add_argument("--destino", default=str(config.RAW_PDF_DIR))
    d.set_defaults(func=cmd_descargar)

    sub.add_parser("estado", help="Muestra el balance del corpus").set_defaults(func=cmd_estado)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
