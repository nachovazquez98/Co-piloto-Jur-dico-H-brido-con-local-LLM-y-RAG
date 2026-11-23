"""Línea de comandos de la evaluación:  python -m copiloto_juridico.evaluacion <comando>

Preparación (sin LLM):
    congelar                         Huella de código, corpus, índice, fichas, configuración, prompts y modelo
    familias                         Exporta evaluacion/familias_penal.csv (tipo -> familia) para revisión
    familias --coherencia            Comprueba la agrupación con los embeddings del índice (v11.2)
    validar                          Valida las consultas, detecta fuga y resume el conjunto
Experimentos sin LLM (usan los conceptos en caché):
    recuperacion --particion P       Líneas base, filtro, k de RRF, k-NN; métricas con IC 95 %
    ablacion --particion P           Quita un componente del clasificador a la vez
    pesos                            Malla de pesos en desarrollo; resultado en prueba una sola vez
Con LLM:
    conceptos                        Precalcula los conceptos clave de cada consulta
    qc --particion P                 QC_1 contra la referencia (inicio vs. fragmento similar)
    latencia --particion P --max N   Tiempos p50/p95 por etapa, llamadas al LLM y descartes
    fichas-batch                     Ficha de cada sentencia penal, una vez (reanudable) + diagnóstico sin anotación
    modelos --max N --fichas M       Compara los modelos de Ollama instalados (latencia, fichas, VRAM)
    robustez                         Consultas difíciles con la app completa (diagnóstico)
    perfil-tribunal --particion P    Las consultas planteadas como toca en estudio frente a la redacción del abogado
Análisis (sin LLM):
    errores --corrida ID [ID ...]    Un caso por fallo, con su tipo (evaluacion/errores/) y casos para la presentación
Referencia automática, sin anotadores humanos (v11.2):
    jueces --pares                   Pares (consulta, sentencia) a juzgar, por agrupamiento (sin LLM)
    jueces --modelo M [--tareas ...] Una tanda de un juez LLM (un modelo a la vez; reanudable)
    referencias                      Reglas + jueces -> referencias, juicios y consultas validadas (sin LLM)
    fichas-referencia --particion P  Fichas precalculadas contra la referencia (sin LLM)
    fundamento-prompt --n N          Prompt de la ficha con el fundamento de la Sala, en desarrollo (con LLM)
    esfuerzo --particion P           Esfuerzo simulado: portal, búsqueda por palabras y app
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .. import config


def _log(msg: str) -> None:
    print(msg, flush=True)


def _guardar(corrida) -> None:
    if "congelado" not in corrida.extra:  # corridas de diagnóstico: solo dejan constancia
        _control(corrida, None, corrida.comando, None)
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


def _control(corrida, particion: str | None, comando: str, motivo: str | None) -> None:
    """Congelado: la partición de prueba exige lo congelado y una sola corrida; desarrollo solo avisa."""
    from .congelado import ErrorCongelado, controlar

    try:
        info = controlar(particion, comando, motivo)
    except ErrorCongelado as e:
        sys.exit(f"❌ {e}")
    for a in info.pop("advertencias"):
        corrida.advertir(a)
    corrida.extra.update(info)


def cmd_congelar(args) -> None:
    from .congelado import congelar
    from .reporte import info_commit

    if info_commit()["sucio"]:
        sys.exit("❌ Hay cambios sin confirmar en src/ o app/: confírmalos antes de congelar (si no, no se puede "
                 "reproducir lo evaluado).")
    datos = congelar()
    _log(f"✅ Congelado {datos['id']} en evaluacion/congelado.json:")
    for k, v in datos.items():
        if k not in ("id", "fecha"):
            _log(f"   {k}: {v}")
    _log("Las corridas con --particion prueba exigirán exactamente esto y correrán una sola vez por comando.")


def cmd_familias(args) -> None:
    from ..datos.corpus import load_corpus
    from .familias import asignar_todas, exportar_csv

    corpus = load_corpus()
    penal = corpus[corpus["materia"] == "penal"]
    if args.coherencia:
        _guardar(_coherencia_familias(penal))
        return
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
    _log(f"\nTabla tipo -> familia: {path}. Una corrección a mano en 'familia' (con 'si' en 'revisado') prevalece "
         "sobre las reglas.")


def _coherencia_familias(penal):
    from ..datos.indice import load_index
    from .familias import cargar_familias, coherencia
    from .reporte import Corrida

    _, chunks, emb = load_index()
    familias = cargar_familias(penal["tipo"].unique().tolist())
    t = coherencia(chunks["doc_id"].to_numpy(), emb, dict(zip(penal["id"], penal["tipo"])), familias)
    corrida = Corrida("familias-coherencia", parametros={"min_sentencias_por_tipo": 3})
    ok = int(t["coincide"].sum()) if len(t) else 0
    corrida.tabla("Resumen", "Tipos con al menos 3 sentencias cuyo centroide de embeddings queda más cerca de su "
                  "propia familia (calculada sin ese tipo) que de cualquier otra. Es una comprobación de "
                  "consistencia sin LLM; no dice qué agrupación es la jurídicamente correcta.",
                  pd.DataFrame([{"tipos": len(t), "más cerca de su familia": ok,
                                 "más cerca de otra": len(t) - ok}]))
    if len(t) - ok:
        corrida.tabla("Tipos más cercanos a otra familia", "Posibles errores de agrupación o tipos con varios delitos.",
                      t[~t["coincide"]].drop(columns="coincide").sort_values("sentencias", ascending=False))
    return corrida


def cmd_borradores(args) -> None:
    from ..datos.corpus import load_corpus
    from .anotacion import redactar_borradores
    from .conjuntos import cargar_consultas, guardar_consultas

    destino = Path(args.consultas)
    textos = load_corpus().set_index("id")["texto"]
    if args.ampliar:  # v11.2: consultas nuevas hasta N por familia, sin tocar las existentes
        from .conjuntos import ampliar_plantilla
        from .familias import cargar_familias

        corpus = load_corpus()
        existentes = cargar_consultas(destino, permitir_vacias=True)
        nuevas = ampliar_plantilla(corpus, cargar_familias(corpus["tipo"].unique().tolist()), existentes,
                                   n_por_familia=args.ampliar)
        _log(f"Redactando {len(nuevas)} consultas nuevas (prompt estricto)...")
        consultas = existentes + redactar_borradores(nuevas, textos, _llm(), intentos=5, estricto=True, log=_log)
    elif args.solo_atencion:  # v11.1: solo las marcadas por el verificador, con el prompt estricto
        consultas = cargar_consultas(destino, permitir_vacias=True)
        from ..pipeline import es_fallo

        marcadas = [c for c in consultas if ("ATENCIÓN" in c.notas or es_fallo(c.consulta)) and not c.revisado]
        _log(f"Regenerando {len(marcadas)} borradores marcados con ATENCIÓN o vacíos (prompt estricto)...")
        redactar_borradores(marcadas, textos, _llm(), intentos=5, estricto=True, log=_log)
    else:
        sys.exit("❌ Indica --ampliar N o --solo-atencion.")
    guardar_consultas(consultas, destino)
    fugas = sum(1 for c in consultas if c.extra.get("fuga"))
    _log(f"\n{len(consultas)} borradores en {destino} (borrador: true, revisado: false); {fugas} aún marcados con "
         "ATENCIÓN. Revísalos y marca revisado: true; la partición de prueba se rechaza mientras haya sin revisar.")


def cmd_validar(args) -> None:
    from ..datos.corpus import load_corpus
    from .conjuntos import (ErrorValidacion, cargar_consultas, detalles_identificables, nombre_tipo_en_consulta,
                            verificar_fuga)
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
        if detalles := detalles_identificables(c.consulta):
            _log(f"⚠️ {c.id} trae datos que identifican la sentencia ({', '.join(detalles[:3])}); quítalos.")
    if fugas:
        sys.exit(1)


def cmd_conceptos(args) -> None:
    from .conjuntos import cargar_consultas
    from .experimentos import calcular_conceptos

    consultas = cargar_consultas(Path(args.consultas), permitir_vacias=True)
    calcular_conceptos(consultas, _llm(con_cache=False), rehacer=args.rehacer, log=_log)


def _entorno(args, comando: str, particion: str | None, **parametros):
    from .experimentos import cargar_entorno
    from .reporte import Corrida

    previo = Corrida(comando)
    _control(previo, particion, comando, getattr(args, "forzar", None))  # antes de cargar nada
    env = cargar_entorno(comando, particion, Path(args.consultas), parametros, log=_log)
    env.corrida.advertencias[:0] = previo.advertencias
    env.corrida.extra.update(previo.extra)
    return env


def cmd_recuperacion(args) -> None:
    from .experimentos import experimento_recuperacion

    env = _entorno(args, "recuperacion", args.particion)
    _guardar(experimento_recuperacion(env, log=_log))


def cmd_esfuerzo(args) -> None:
    from .esfuerzo import experimento_esfuerzo

    env = _entorno(args, "esfuerzo", args.particion)
    _guardar(experimento_esfuerzo(env, log=_log))


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
        _control(env_dev.corrida, "prueba", "pesos", args.forzar)
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


def cmd_fichas_batch(args) -> None:
    from ..datos.corpus import load_corpus
    from ..fichas_precalculadas import actualizar_verificaciones, cargar, generar_todas, version_ficha
    from .experimentos import diagnostico_fichas

    settings = config.LLMSettings()
    corpus = load_corpus()
    if not args.solo_diagnostico:
        from .jueces import en_muestra

        # --muestra: solo las sentencias de la muestra de la referencia (las únicas que evalúa fichas-referencia);
        # las demás las genera la app en el momento si hacen falta.
        lote = corpus[[en_muestra(d, config.MUESTRA_SENTENCIAS) for d in corpus["id"]]] if args.muestra else corpus
        llm = _llm(con_cache=False)
        if args.respaldo:
            # --respaldo: con el modelo de respaldo, solo las sentencias sin ficha válida del principal y fuera de la
            # muestra de la referencia (la evaluación mide únicamente las fichas del modelo principal).
            import dataclasses

            from ..llm import get_client
            from .modelos import descargar_de_memoria

            if not config.FICHAS_MODELO_RESPALDO:
                sys.exit("❌ No hay modelo de respaldo (config.FICHAS_MODELO_RESPALDO).")
            tabla = cargar()
            principal = version_ficha(settings)
            con_ficha = set(tabla.loc[(tabla["version"] == principal) & tabla["json_valido"].astype(bool), "doc_id"])
            lote = lote[~lote["id"].isin(con_ficha) & ~lote["id"].map(lambda d: en_muestra(d, config.MUESTRA_SENTENCIAS))]
            descargar_de_memoria(settings)  # nunca dos modelos cargados a la vez
            llm = get_client(dataclasses.replace(settings, model=config.FICHAS_MODELO_RESPALDO))
            _log(f"Respaldo {config.FICHAS_MODELO_RESPALDO}: {int((lote['materia'] == 'penal').sum())} sentencias "
                 "penales sin ficha del modelo principal, fuera de la muestra.")
        generar_todas(lote, llm, limite=args.max, log=_log)
        if args.respaldo:
            descargar_de_memoria(llm.settings)
    _log(f"Verificaciones recalculadas con las reglas actuales: {actualizar_verificaciones(corpus)} fichas cambiaron.")
    _guardar(diagnostico_fichas(cargar(), version_ficha(settings)))


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


def cmd_modelos(args) -> None:
    from ..clasificador import construir_clasificador
    from ..vectorstore import SentenciasStore
    from .modelos import comparar

    store = SentenciasStore(log=lambda _m: None)
    clas = construir_clasificador(store.corpus["tipo"].unique().tolist())
    _guardar(comparar(store, clas, args.modelos or config.MODELOS_COMPARACION, args.max, args.fichas, log=_log))


def cmd_robustez(args) -> None:
    from ..clasificador import construir_clasificador
    from ..vectorstore import SentenciasStore
    from .robustez import correr

    store = SentenciasStore(log=lambda _m: None)
    clas = construir_clasificador(store.corpus["tipo"].unique().tolist())
    # sin la caché de respuestas del LLM: guardaría en disco los prompts con los datos personales inventados
    _guardar(correr(store, clas, _llm(con_cache=False), log=_log))


def cmd_jueces(args) -> None:
    from . import jueces as J

    if args.pares:
        J.guardar_pares(log=_log)
        return
    if not args.modelo:
        sys.exit("❌ Indica --modelo (uno de config.MODELOS_JUECES) o --pares.")
    for r in J.correr(args.modelo, tuple(args.tareas or J.TAREAS), args.max, log=_log, completa=args.completa):
        _log(f"   {r}")


def cmd_referencias(args) -> None:
    from .referencias import construir

    _guardar(construir(args.modelos, log=_log, completa=args.completa))


def cmd_fichas_referencia(args) -> None:
    from .referencias import evaluar_fichas

    corrida = evaluar_fichas(args.particion, log=_log)
    _control(corrida, args.particion if args.particion != "desarrollo" else None, "fichas-referencia", args.forzar)
    _guardar(corrida)


def cmd_perfil_tribunal(args) -> None:
    from .experimentos import experimento_perfil_tribunal

    env = _entorno(args, "perfil-tribunal", args.particion)
    _guardar(experimento_perfil_tribunal(env, _llm(), log=_log))


def cmd_fundamento_prompt(args) -> None:
    from .referencias import experimento_fundamento

    _guardar(experimento_fundamento(args.n, _llm(), log=_log))


def cmd_errores(args) -> None:
    from ..clasificador import construir_clasificador
    from ..vectorstore import SentenciasStore
    from .errores import ejecutar

    store = SentenciasStore(log=lambda _m: None)
    clas = construir_clasificador(store.corpus["tipo"].unique().tolist())
    try:
        escritos = ejecutar(args.corrida, store, clas, log=_log)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"❌ {e}")
    _log("Reportes:\n" + "\n".join(f"  {p}" for p in escritos))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config.CACHE_RESULTADOS = False  # la evaluación nunca usa resultados guardados de la app
    p = argparse.ArgumentParser(prog="python -m copiloto_juridico.evaluacion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--consultas", default=str(config.CONSULTAS_PATH), help="JSONL de consultas")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("congelar", help="Guarda la huella de lo que se evalúa (evaluacion/congelado.json)")
    s.set_defaults(func=cmd_congelar)

    s = sub.add_parser("familias", help="Exporta el CSV tipo -> familia para revisión")
    s.add_argument("--forzar", action="store_true", help="Sobrescribe el CSV (se pierde la revisión)")
    s.add_argument("--coherencia", action="store_true", help="Comprueba la agrupación con embeddings (sin LLM)")
    s.set_defaults(func=cmd_familias)

    s = sub.add_parser("borradores", help="Borradores de consultas con el LLM (verificador de fuga incluido)")
    s.add_argument("--ampliar", type=int, metavar="N", help="Agrega consultas hasta N por familia (mitad por partición)")
    s.add_argument("--solo-atencion", action="store_true",
                   help="Regenera solo los marcados con ATENCIÓN (o que no son una consulta), con el prompt estricto")
    s.set_defaults(func=cmd_borradores)

    s = sub.add_parser("validar", help="Valida las consultas y detecta fuga")
    s.add_argument("--permitir-vacias", action="store_true", help="Acepta consultas aún sin redactar")
    s.set_defaults(func=cmd_validar)

    s = sub.add_parser("conceptos", help="Precalcula conceptos clave con el LLM")
    s.add_argument("--rehacer", action="store_true")
    s.set_defaults(func=cmd_conceptos)

    for nombre, func, ayuda in (("recuperacion", cmd_recuperacion, "Experimentos de recuperación y clasificación"),
                                ("ablacion", cmd_ablacion, "Ablación del clasificador"),
                                ("qc", cmd_qc, "QC_1 contra la referencia automática"),
                                ("latencia", cmd_latencia, "Latencia de punta a punta"),
                                ("esfuerzo", cmd_esfuerzo, "Sentencias que hay que abrir para encontrar 3 útiles")):
        s = sub.add_parser(nombre, help=ayuda)
        s.add_argument("--particion", choices=["desarrollo", "prueba", "todas"], default="desarrollo")
        s.add_argument("--forzar", metavar="MOTIVO", help="Repite una corrida de prueba ya hecha (queda en el reporte)")
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
    s.add_argument("--forzar", metavar="MOTIVO", help="Repite una corrida de prueba ya hecha (queda en el reporte)")
    s.set_defaults(func=cmd_pesos)

    s = sub.add_parser("fichas-batch", help="Ficha JSON de cada sentencia penal, una sola vez (reanudable)")
    s.add_argument("--max", type=int, default=None, help="Fichas a generar en esta corrida (todas por defecto)")
    s.add_argument("--solo-diagnostico", action="store_true", help="Solo el reporte, sin llamar al LLM")
    s.add_argument("--muestra", action="store_true",
                   help="Solo las sentencias de la muestra de la referencia (config.MUESTRA_SENTENCIAS)")
    s.add_argument("--respaldo", action="store_true",
                   help="Con config.FICHAS_MODELO_RESPALDO, las que no tienen ficha del modelo principal (fuera de la muestra)")
    s.set_defaults(func=cmd_fichas_batch)

    s = sub.add_parser("qc-diagnostico", help="Tasa de descarte de QC_1 por versión y contexto (sin anotaciones)")
    s.add_argument("--k", type=int, default=5, help="Candidatos por consulta")
    s.set_defaults(func=cmd_qc_diagnostico)

    s = sub.add_parser("regresion", help="Top-3 de categorías en los ejemplos fijos (sin LLM)")
    s.add_argument("--recalcular-conceptos", action="store_true", help="Precalcula los conceptos con el LLM")
    s.set_defaults(func=cmd_regresion)

    s = sub.add_parser("modelos", help="Compara los modelos de Ollama ya instalados (no descarga ninguno)")
    s.add_argument("--max", type=int, default=10, help="Ejemplos penales para la latencia")
    s.add_argument("--fichas", type=int, default=20, help="Sentencias penales (las mismas para cada modelo)")
    s.add_argument("--modelos", nargs="+", help="Lista de modelos (por defecto config.MODELOS_COMPARACION)")
    s.set_defaults(func=cmd_modelos)

    s = sub.add_parser("robustez", help="Consultas difíciles con el LLM real (diagnóstico)")
    s.set_defaults(func=cmd_robustez)

    s = sub.add_parser("errores", help="Análisis de errores de corridas guardadas (sin LLM)")
    s.add_argument("--corrida", nargs="+", required=True, help="Nombre de la corrida en evaluacion/resultados/")
    s.set_defaults(func=cmd_errores)

    s = sub.add_parser("jueces", help="Jueces LLM de la evaluación automática (una tanda por modelo)")
    s.add_argument("--pares", action="store_true", help="Solo arma evaluacion/jueces/pares.csv (sin LLM)")
    s.add_argument("--modelo", help="Modelo de Ollama ya instalado (p. ej. qwen2.5:7b)")
    s.add_argument("--tareas", nargs="+", choices=["sentencia", "relevancia", "consulta"])
    s.add_argument("--max", type=int, default=None, help="Elementos por tarea en esta tanda (todos por defecto)")
    s.add_argument("--completa", action="store_true", help="Todo el corpus en lugar de la muestra (config.MUESTRA_*)")
    s.set_defaults(func=cmd_jueces)

    s = sub.add_parser("referencias", help="Referencias automáticas a partir de reglas y jueces (sin LLM)")
    s.add_argument("--modelos", nargs="+", help="Jueces (por defecto config.MODELOS_JUECES)")
    s.add_argument("--completa", action="store_true", help="Todo el corpus en lugar de la muestra de los jueces")
    s.set_defaults(func=cmd_referencias)

    s = sub.add_parser("fichas-referencia", help="Fichas precalculadas contra la referencia automática (sin LLM)")
    s.add_argument("--particion", choices=["desarrollo", "prueba", "todas"], default="desarrollo")
    s.add_argument("--forzar", metavar="MOTIVO", help="Repite una corrida de prueba ya hecha (queda en el reporte)")
    s.set_defaults(func=cmd_fichas_referencia)

    s = sub.add_parser("perfil-tribunal", help="Consultas como toca en estudio frente a la redacción del abogado (LLM)")
    s.add_argument("--particion", choices=["desarrollo", "prueba", "todas"], default="desarrollo")
    s.add_argument("--consultas", default=str(config.CONSULTAS_PATH))
    s.set_defaults(func=cmd_perfil_tribunal)

    s = sub.add_parser("fundamento-prompt", help="Prompt de la ficha con el fundamento de la Sala (desarrollo, con LLM)")
    s.add_argument("--n", type=int, default=120, help="Sentencias de desarrollo de la muestra estable por hash")
    s.set_defaults(func=cmd_fundamento_prompt)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
