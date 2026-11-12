"""Línea de comandos de la evaluación:  python -m copiloto_juridico.evaluacion <comando>

Preparación (sin LLM):
    familias                         Exporta evaluacion/familias_penal.csv (tipo -> familia) para revisión
    plantilla                        Borrador estratificado de consultas (evaluacion/consultas_penal_borrador.jsonl)
    validar                          Valida consultas y juicios, detecta fuga y resume el conjunto
Experimentos sin LLM (usan los conceptos en caché):
    recuperacion --particion P       Líneas base, filtro, k de RRF, k-NN; métricas con IC 95 %
    ablacion --particion P           Quita un componente del clasificador a la vez
    pesos                            Malla de pesos en desarrollo; resultado en prueba una sola vez
Con LLM:
    conceptos                        Precalcula los conceptos clave de cada consulta
    qc --particion P                 QC_1 vs. juicios humanos (inicio vs. fragmento similar) y QC_2
    generacion                       Fichas del conjunto C vs. anotación; CSV para marcar alucinaciones
    latencia --particion P --max N   Tiempos p50/p95 por etapa, llamadas al LLM y descartes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .. import config


def _log(msg: str) -> None:
    print(msg, flush=True)


def _guardar(corrida) -> None:
    pj, pm = corrida.guardar()
    print()
    print(corrida.markdown())
    _log(f"\nResultados guardados en:\n  {pj}\n  {pm}")
    if corrida.borrador:
        _log("⚠️  Corrida con consultas en BORRADOR: no reportable.")


def _llm(con_cache: bool = True):
    from ..llm import get_client
    from .experimentos import LLMConCache

    settings = config.LLMSettings()
    cliente = get_client(settings)
    ok, msg = cliente.health()
    if not ok:
        sys.exit(f"❌ {msg}")
    _log(f"LLM: {msg} · semilla {getattr(settings, 'seed', None)} · temperatura {settings.temperature}")
    return LLMConCache(cliente) if con_cache else cliente


def cmd_familias(args) -> None:
    from ..datos.corpus import load_corpus
    from .familias import asignar_todas, exportar_csv

    corpus = load_corpus()
    penal = corpus[corpus["materia"] == "penal"]
    conteo = penal["tipo"].value_counts().to_dict()
    try:
        path = exportar_csv(conteo, forzar=args.forzar)
    except FileExistsError as e:
        sys.exit(f"❌ {e}")
    asig = asignar_todas(list(conteo))
    por_fam = Counter()
    for a in asig:
        por_fam[a.familia] += conteo[a.tipo]
    _log(f"{len(asig)} tipos penales -> {len(por_fam)} familias. Para revisar: {sum(a.revisar for a in asig)}.")
    for f, n in por_fam.most_common():
        _log(f"   {n:5d}  {f}")
    _log(f"\nCSV para el abogado: {path}\nEdita la columna 'familia' y pon 'si' en 'revisado'. "
         "El código usará este archivo en lugar de las reglas.")


def cmd_plantilla(args) -> None:
    from ..datos.corpus import load_corpus
    from .conjuntos import guardar_consultas, muestrear_plantilla
    from .familias import cargar_familias

    corpus = load_corpus()
    familias = cargar_familias(corpus["tipo"].unique().tolist())
    plantilla = muestrear_plantilla(corpus, familias, n_por_familia=args.n_por_familia, n_raros=args.n_raros,
                                    n_sin_respuesta=args.n_sin_respuesta, semilla=args.semilla)
    path = Path(args.salida)
    if path.resolve() == config.CONSULTAS_PATH.resolve():
        sys.exit("❌ La plantilla no puede sobrescribir el conjunto final; usa otra ruta de salida.")
    guardar_consultas(plantilla, path)
    dist = Counter((c.familia or "sin respuesta", c.particion) for c in plantilla)
    _log(f"\nPlantilla con {len(plantilla)} elementos (borrador: true) en {path}")
    _log(pd.Series(dist).unstack(fill_value=0).to_string())
    _log("\nSiguiente paso: redacta/corrige cada 'consulta', pon borrador: false, guarda como "
         f"{config.CONSULTAS_PATH.name} y corre `validar`.")


def cmd_borradores(args) -> None:
    from ..datos.corpus import load_corpus
    from .anotacion import redactar_borradores
    from .conjuntos import cargar_consultas, guardar_consultas

    destino = Path(args.consultas)
    if destino.exists() and not args.forzar:
        sys.exit(f"❌ {destino} ya existe (puede tener consultas revisadas). Usa --forzar para sobrescribirlo.")
    plantilla = cargar_consultas(Path(args.plantilla), permitir_vacias=True)
    textos = load_corpus().set_index("id")["texto"]
    consultas = redactar_borradores(plantilla, textos, _llm(), n=args.n, log=_log)
    guardar_consultas(consultas, destino)
    fugas = sum(1 for c in consultas if c.extra.get("fuga"))
    _log(f"\n{len(consultas)} borradores en {destino} (borrador: true, revisado: false); {fugas} aún con fuga. "
         "Revísalos y marca revisado: true; la partición de prueba se rechaza mientras haya sin revisar.")


def cmd_hojas(args) -> None:
    from ..clasificador import construir_clasificador
    from ..vectorstore import SentenciasStore
    from .anotacion import hoja_fichas, hoja_juicios
    from .conjuntos import cargar_consultas
    from .experimentos import conceptos_de, leer_conceptos
    from .familias import cargar_familias

    for path in (config.JUICIOS_PATH, config.FICHAS_ANOTADAS_PATH):
        if path.exists() and not args.forzar:
            sys.exit(f"❌ {path} ya existe (puede tener anotaciones). Usa --forzar para sobrescribirlo.")
    store = SentenciasStore(log=lambda _m: None)
    tipos = store.corpus["tipo"].unique().tolist()
    consultas = cargar_consultas(Path(args.consultas), permitir_vacias=True)
    conceptos, _ = conceptos_de(consultas, leer_conceptos())
    jz = hoja_juicios(store, construir_clasificador(tipos), consultas, conceptos, n_consultas=args.n_consultas)
    jz.to_csv(config.JUICIOS_PATH, index=False)
    fc = hoja_fichas(store.corpus, cargar_familias(tipos), n=args.n_fichas)
    fc.to_csv(config.FICHAS_ANOTADAS_PATH, index=False)
    _log(f"Hoja de juicios: {config.JUICIOS_PATH} ({jz['consulta_id'].nunique()} consultas, {len(jz)} pares)\n"
         f"Hoja de fichas: {config.FICHAS_ANOTADAS_PATH} ({len(fc)} sentencias). Instrucciones en evaluacion/README.md.")


def cmd_validar(args) -> None:
    from ..datos.corpus import load_corpus
    from .conjuntos import (ErrorValidacion, acuerdo_entre_anotadores, cargar_consultas, cargar_juicios,
                            nombre_tipo_en_consulta, verificar_fuga)
    from .familias import cargar_familias

    corpus = load_corpus()
    familias = cargar_familias(corpus["tipo"].unique().tolist())
    try:
        consultas = cargar_consultas(Path(args.consultas), corpus, familias, permitir_vacias=args.permitir_vacias)
    except (ErrorValidacion, FileNotFoundError) as e:
        sys.exit(f"❌ {e}")
    _log(f"✅ {len(consultas)} consultas válidas.")
    _log(f"   Borrador: {sum(c.borrador for c in consultas)} · sin redactar: "
         f"{sum(not c.consulta.strip() for c in consultas)} · sin respuesta: {sum(c.sin_respuesta for c in consultas)}"
         f" · documento conocido: {sum(c.documento_conocido for c in consultas)}")
    dist = Counter((c.familia or ("sin respuesta" if c.sin_respuesta else "conjunto B"), c.particion)
                   for c in consultas)
    _log(pd.Series(dist).unstack(fill_value=0).to_string())
    fugas = verificar_fuga(consultas, corpus)
    for cid, grams in fugas.items():
        _log(f"⚠️ Fuga en {cid}: {len(grams)} secuencias de 7+ palabras, p. ej. «{grams[0]}»")
    for c in consultas:
        if nombre_tipo_en_consulta(c):
            _log(f"⚠️ {c.id} menciona literalmente el nombre del tipo ({c.tipo}); facilita la clasificación.")
    try:
        juicios = cargar_juicios(config.JUICIOS_PATH, {c.id for c in consultas})
    except ErrorValidacion as e:
        sys.exit(f"❌ {e}")
    if len(juicios):
        _log(f"✅ {len(juicios)} juicios de relevancia de {juicios['anotador'].nunique()} anotador(es).")
        acuerdo = acuerdo_entre_anotadores(juicios)
        if acuerdo:
            _log(f"   Acuerdo {acuerdo['anotadores']}: kappa = {acuerdo['kappa_0_1_2']:.3f} (0/1/2), "
                 f"{acuerdo['kappa_binario']:.3f} (binario) en {acuerdo['pares']} pares"
                 + ("  ⚠️ < 0.6: revisa la guía de anotación" if acuerdo["kappa_0_1_2"] < 0.6 else ""))
    else:
        _log("ℹ️ Sin juicios de relevancia todavía (juicios.csv).")
    if fugas:
        sys.exit(1)


def cmd_conceptos(args) -> None:
    from .conjuntos import cargar_consultas
    from .experimentos import calcular_conceptos

    consultas = cargar_consultas(Path(args.consultas), permitir_vacias=True)
    calcular_conceptos(consultas, _llm(con_cache=False), rehacer=args.rehacer, log=_log)


def _entorno(args, comando: str, particion: str | None, **parametros):
    from .experimentos import cargar_entorno

    return cargar_entorno(comando, particion, Path(args.consultas), parametros, log=_log)


def cmd_recuperacion(args) -> None:
    from .experimentos import experimento_recuperacion

    env = _entorno(args, "recuperacion", args.particion)
    _guardar(experimento_recuperacion(env, log=_log))


def cmd_ablacion(args) -> None:
    from .experimentos import experimento_ablacion

    env = _entorno(args, "ablacion", args.particion)
    _guardar(experimento_ablacion(env, log=_log))


def cmd_pesos(args) -> None:
    from .experimentos import experimento_pesos

    env_dev = _entorno(args, "pesos", "desarrollo")
    env_prueba = None
    if not args.sin_prueba:
        from .conjuntos import cargar_consultas

        from .conjuntos import ErrorValidacion, exigir_revisadas

        prueba = [c for c in cargar_consultas(Path(args.consultas), permitir_vacias=True)
                  if c.particion == "prueba" and c.consulta.strip()]
        try:
            exigir_revisadas(prueba)
        except ErrorValidacion as e:
            sys.exit(f"❌ {e} (o usa --sin-prueba)")
        env_prueba = env_dev.__class__(env_dev.store, env_dev.clasificador, env_dev.familias, prueba,
                                       {**env_dev.conceptos}, env_dev.juicios, env_dev.corrida)
        from .experimentos import conceptos_de, leer_conceptos

        env_prueba.conceptos, _ = conceptos_de(prueba, leer_conceptos())
    _guardar(experimento_pesos(env_dev, env_prueba, objetivo=args.objetivo, paso=args.paso, log=_log))


def cmd_qc(args) -> None:
    from .experimentos import experimento_qc

    env = _entorno(args, "qc", args.particion)
    variantes = ["inicio", "fragmento_similar"] if args.variante == "ambas" else [args.variante]
    _guardar(experimento_qc(env, _llm(), variantes, log=_log))


def cmd_generacion(args) -> None:
    from ..datos.corpus import load_corpus
    from .conjuntos import cargar_fichas_anotadas
    from .experimentos import experimento_generacion
    from .reporte import Corrida

    fichas = cargar_fichas_anotadas()
    if not fichas:
        sys.exit(f"❌ No hay fichas anotadas en {config.FICHAS_ANOTADAS_PATH} (ver evaluacion/README.md).")
    corrida = Corrida("generacion")
    corrida, muestra, generadas = experimento_generacion(fichas, load_corpus(), _llm(), corrida,
                                                         version_prompt=args.prompt_penal,
                                                         n_afirmaciones=args.n_afirmaciones, log=_log)
    pj, _ = corrida.guardar()
    base = pj.with_suffix("")
    muestra.to_csv(f"{base}_afirmaciones.csv", index=False)
    with open(f"{base}_fichas.jsonl", "w", encoding="utf-8") as f:
        for g in generadas:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(corrida.markdown())
    _log(f"\nResultados: {pj}\nAfirmaciones para marcar alucinaciones: {base}_afirmaciones.csv")


def cmd_qc_diagnostico(args) -> None:
    from ..clasificador import construir_clasificador
    from ..vectorstore import SentenciasStore
    from .experimentos import experimento_qc_diagnostico

    store = SentenciasStore(log=lambda _m: None)
    clas = construir_clasificador(store.corpus["tipo"].unique().tolist())
    _guardar(experimento_qc_diagnostico(store, clas, _llm(), k=args.k, log=_log))


def cmd_regresion(args) -> None:
    from . import regresion

    if args.recalcular_conceptos:
        regresion.calcular_conceptos(_llm(con_cache=False), log=_log)
    from ..vectorstore import SentenciasStore

    _log("Cargando la base de sentencias...")
    _guardar(regresion.correr_regresion(SentenciasStore(log=lambda _m: None), log=_log))


def cmd_latencia(args) -> None:
    from .experimentos import experimento_latencia

    if args.ejemplos:  # los ejemplos penales de la app (no necesita consultas redactadas)
        from ..clasificador import construir_clasificador
        from ..vectorstore import SentenciasStore
        from .conjuntos import Consulta
        from .experimentos import Entorno
        from .regresion import EJEMPLOS_PENALES
        from .reporte import Corrida

        store = SentenciasStore(log=lambda _m: None)
        consultas = [Consulta(f"E{i:02d}", e.texto, None, None, None, "desarrollo")
                     for i, e in enumerate(EJEMPLOS_PENALES, 1)]
        corrida = Corrida("latencia", parametros={"consultas": "ejemplos penales de la app",
                                                  "ficha": config.FICHA_MODO, "qc2": config.QC2_MODO})
        env = Entorno(store, construir_clasificador(store.corpus["tipo"].unique().tolist()), {}, consultas, {}, {},
                      corrida)
    else:
        env = _entorno(args, "latencia", args.particion)
    _guardar(experimento_latencia(env, _llm(con_cache=False), max_consultas=args.max, log=_log))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    p = argparse.ArgumentParser(prog="python -m copiloto_juridico.evaluacion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--consultas", default=str(config.CONSULTAS_PATH), help="JSONL de consultas")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("familias", help="Exporta el CSV tipo -> familia para revisión")
    s.add_argument("--forzar", action="store_true", help="Sobrescribe el CSV (se pierde la revisión)")
    s.set_defaults(func=cmd_familias)

    s = sub.add_parser("plantilla", help="Borrador estratificado de consultas a redactar")
    s.add_argument("--n-por-familia", type=int, default=5,
                   help="Sentencias por familia con >= 10 sentencias (5 x 20 familias + 20 raros = 120)")
    s.add_argument("--n-raros", type=int, default=20)
    s.add_argument("--n-sin-respuesta", type=int, default=10)
    s.add_argument("--semilla", type=int, default=config.EVAL_SEED)
    s.add_argument("--salida", default=str(config.EVAL_DIR / "consultas_penal_borrador.jsonl"))
    s.set_defaults(func=cmd_plantilla)

    s = sub.add_parser("borradores", help="Borradores de consultas con el LLM (verificador de fuga incluido)")
    s.add_argument("--n", type=int, default=None, help="Entradas de la plantilla a redactar (todas por defecto)")
    s.add_argument("--plantilla", default=str(config.EVAL_DIR / "consultas_penal_borrador.jsonl"))
    s.add_argument("--forzar", action="store_true")
    s.set_defaults(func=cmd_borradores)

    s = sub.add_parser("hojas-anotacion", help="Hojas CSV para el abogado (juicios y fichas)")
    s.add_argument("--n-consultas", type=int, default=30)
    s.add_argument("--n-fichas", type=int, default=40)
    s.add_argument("--forzar", action="store_true")
    s.set_defaults(func=cmd_hojas)

    s = sub.add_parser("validar", help="Valida consultas/juicios y detecta fuga")
    s.add_argument("--permitir-vacias", action="store_true", help="Acepta consultas aún sin redactar")
    s.set_defaults(func=cmd_validar)

    s = sub.add_parser("conceptos", help="Precalcula conceptos clave con el LLM")
    s.add_argument("--rehacer", action="store_true")
    s.set_defaults(func=cmd_conceptos)

    for nombre, func, ayuda in (("recuperacion", cmd_recuperacion, "Experimentos de recuperación y clasificación"),
                                ("ablacion", cmd_ablacion, "Ablación del clasificador"),
                                ("qc", cmd_qc, "Controles de calidad vs. juicio humano"),
                                ("latencia", cmd_latencia, "Latencia de punta a punta")):
        s = sub.add_parser(nombre, help=ayuda)
        s.add_argument("--particion", choices=["desarrollo", "prueba", "todas"], default="desarrollo")
        if nombre == "qc":
            s.add_argument("--variante", choices=["inicio", "fragmento_similar", "ambas"], default="ambas")
        if nombre == "latencia":
            s.add_argument("--max", type=int, default=10, help="Número de consultas a ejecutar")
            s.add_argument("--ejemplos", action="store_true", help="Usa los ejemplos penales de la app")
        s.set_defaults(func=func)

    s = sub.add_parser("pesos", help="Malla de pesos en desarrollo, resultado en prueba una vez")
    s.add_argument("--objetivo", choices=["hit3_familia", "hit3_tipo", "top1_familia", "top1_tipo"],
                   default="hit3_familia")
    s.add_argument("--paso", type=float, default=0.05)
    s.add_argument("--sin-prueba", action="store_true", help="Solo desarrollo (no toca la partición de prueba)")
    s.set_defaults(func=cmd_pesos)

    s = sub.add_parser("qc-diagnostico", help="Tasa de descarte de QC_1 por versión y contexto (sin anotaciones)")
    s.add_argument("--k", type=int, default=5, help="Candidatos por consulta")
    s.set_defaults(func=cmd_qc_diagnostico)

    s = sub.add_parser("regresion", help="Top-3 de categorías en los ejemplos fijos (sin LLM)")
    s.add_argument("--recalcular-conceptos", action="store_true", help="Precalcula los conceptos con el LLM")
    s.set_defaults(func=cmd_regresion)

    s = sub.add_parser("generacion", help="Fichas del conjunto C vs. anotación")
    s.add_argument("--prompt-penal", choices=["v10", "v11"], default=None)
    s.add_argument("--n-afirmaciones", type=int, default=100)
    s.set_defaults(func=cmd_generacion)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
