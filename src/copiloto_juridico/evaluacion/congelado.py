"""Congelar lo que se evalúa.

`congelar` guarda en `evaluacion/congelado.json` la huella de todo lo que cambia los resultados: código, corpus,
índice, fichas precalculadas, configuración efectiva (config.py + variables de entorno), texto de cada prompt y
digest del modelo de Ollama. Después:

- Una corrida con la partición de **prueba** (o "todas", que la incluye) se niega a correr si algo no coincide, y
  solo puede correr **una vez** por congelado y comando; repetirla exige `--forzar "motivo"`, que queda en el reporte.
- Una corrida de desarrollo solo avisa.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .. import config, prompts

CONGELADO_PATH = config.EVAL_DIR / "congelado.json"
REGISTRO_PRUEBA_PATH = config.EVAL_DIR / "prueba_corridas.jsonl"
PARTICIONES_FINALES = ("prueba", "todas")


class ErrorCongelado(RuntimeError):
    """La corrida final no puede hacerse (el mensaje cabe en una línea)."""


def _sha(*partes: bytes) -> str:
    h = hashlib.sha256()
    for p in partes:
        h.update(p)
    return h.hexdigest()[:16]


def _archivos(*patrones: tuple[Path, str]) -> str:
    rutas = sorted(p for base, patron in patrones for p in base.glob(patron) if p.is_file())
    return _sha(*(str(p.relative_to(config.ROOT_DIR)).encode() + p.read_bytes() for p in rutas)) if rutas else "—"


def huella_codigo() -> str:
    """Contenido de todos los .py de src/ y app/ (no depende de git ni de si hay cambios sin confirmar)."""
    return _archivos((config.ROOT_DIR / "src", "**/*.py"), (config.ROOT_DIR / "app", "**/*.py"))


def huella_corpus() -> str:
    from ..datos.corpus import load_corpus

    c = load_corpus()
    return _sha(*(f"{d}\x00{t}\x00".encode() for d, t in zip(c["id"], c["texto"])))


def huella_indice() -> str:
    return _archivos((config.INDICE_DIR, "*"))


def huella_fichas() -> str:
    p = config.FICHAS_PRECALCULADAS_PATH
    return _sha(p.read_bytes()) if p.exists() and config.FICHAS_PRECALCULADAS else "sin fichas precalculadas"


def hash_configuracion() -> str:
    from .reporte import snapshot_config

    return _sha(json.dumps(snapshot_config(), sort_keys=True, default=str).encode())


def versiones_prompts() -> dict[str, str]:
    """Versión configurada y hash del texto de cada prompt que usa la app (editar un prompt sin cambiar su
    versión también cuenta como cambio)."""
    ext, rep = prompts.prompts_penal()
    usados = {
        "conceptos": (config.PROMPT_CONCEPTOS_VERSION, prompts.prompt_conceptos()),
        "qc1": (config.PROMPT_QC1_VERSION, prompts.prompt_qc1()),
        "ficha_json": (config.FICHA_MODO, prompts.PROMPT_FICHA_PENAL_JSON),
        "penal_dos_llamadas": (config.PROMPT_PENAL_VERSION, ext + rep),
        "qc2_llm": (config.QC2_MODO, prompts.PROMPT_VERIFICACION_POST_REPORTE),
        "sintesis": (config.PROMPT_SINTESIS_VERSION, prompts.prompt_sintesis()),
        # los de los jueces de la evaluación automática (no los usa la app, pero cambian la referencia)
        "juez_sentencia": ("v11.2", prompts.PROMPT_JUEZ_SENTENCIA),
        "juez_relevancia": ("v11.2", prompts.PROMPT_JUEZ_RELEVANCIA),
        "juez_consulta": ("v11.2", prompts.PROMPT_JUEZ_CONSULTA),
    }
    return {k: f"{v} · {_sha(t.encode())}" for k, (v, t) in usados.items()}


def estado_actual() -> dict:
    from ..llm import digest_modelo

    s = config.LLMSettings()
    return {"codigo": huella_codigo(), "corpus": huella_corpus(), "indice": huella_indice(),
            "fichas_precalculadas": huella_fichas(), "configuracion": hash_configuracion(),
            "prompts": versiones_prompts(), "modelo": s.model, "digest_modelo": digest_modelo(s) or "no disponible"}


def identificador(estado: dict) -> str:
    return _sha(json.dumps(estado, sort_keys=True).encode())


def congelar(path: Path = CONGELADO_PATH) -> dict:
    estado = estado_actual()
    datos = {"id": identificador(estado), "fecha": datetime.now().isoformat(timespec="seconds"), **estado}
    path.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return datos


def leer(path: Path = CONGELADO_PATH) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def diferencias(congelado: dict, actual: dict) -> list[str]:
    """Qué cambió desde el congelado (nombres de componente; los prompts, uno por uno)."""
    out = [k for k in actual if k != "prompts" and congelado.get(k) != actual[k]]
    out += [f"prompt {k}" for k, v in actual["prompts"].items() if congelado.get("prompts", {}).get(k) != v]
    return out


def controlar(particion: str | None, comando: str, motivo_forzar: str | None = None,
              path: Path = CONGELADO_PATH, registro: Path = REGISTRO_PRUEBA_PATH) -> dict:
    """Antes de una corrida. Prueba: exige congelado vigente y que sea la primera vez (o un motivo para repetir), y
    deja la marca. Desarrollo: devuelve las diferencias como advertencias. Devuelve lo que va al reporte."""
    congelado = leer(path)
    final = particion in PARTICIONES_FINALES
    if congelado is None:
        if final:
            raise ErrorCongelado("No hay evaluacion/congelado.json: corre `congelar` antes de usar la partición de prueba.")
        return {"congelado": None, "advertencias": ["Sin congelado: corrida de desarrollo no atada a una versión."]}
    cambios = diferencias(congelado, estado_actual())
    if final and cambios:
        raise ErrorCongelado(f"Lo evaluado no coincide con congelado.json ({', '.join(cambios)}): vuelve a congelar "
                             "o restaura lo congelado.")
    info = {"congelado": congelado["id"], "advertencias": [f"Distinto de lo congelado: {', '.join(cambios)}."]
            if cambios else []}
    if not final:
        return info
    previas = [json.loads(x) for x in registro.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if registro.exists() else []
    previas = [p for p in previas if p["congelado"] == congelado["id"] and p["comando"] == comando]
    if previas and not motivo_forzar:
        raise ErrorCongelado(f"La partición de prueba ya se usó para `{comando}` con este congelado "
                             f"({previas[-1]['fecha']}). Para repetir: --forzar \"motivo\".")
    info["motivo_forzar"] = motivo_forzar if previas else None
    with open(registro, "a", encoding="utf-8") as f:
        f.write(json.dumps({"congelado": congelado["id"], "comando": comando, "particion": particion,
                            "fecha": datetime.now().isoformat(timespec="seconds"), "motivo": info["motivo_forzar"]},
                           ensure_ascii=False) + "\n")
    return info
