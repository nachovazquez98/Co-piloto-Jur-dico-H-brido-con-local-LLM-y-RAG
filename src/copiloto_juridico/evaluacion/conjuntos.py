"""Conjuntos de evaluación: consultas y juicios de relevancia.

- `consultas_penal.jsonl`, una consulta por línea:
  `{"id", "consulta", "doc_id_fuente", "tipo", "familia", "particion", "sin_respuesta", "borrador"?, "notas"?}`
  * Documento conocido: `doc_id_fuente` es la sentencia de la que se redactó la consulta (`borradores`).
  * Sin respuesta: `sin_respuesta: true`, tema sin tipo en el corpus; el sistema debería reconocerlo.
- Juicios de relevancia (`consulta_id, doc_id, relevancia 0/1/2, anotador`): los escribe `referencias` en
  `evaluacion/jueces/juicios_automaticos.csv`.

Solo cuentan las consultas validadas por `referencias` (comprobaciones de la guía y el juez LLM).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

PARTICIONES = ("desarrollo", "prueba")
N_GRAMA_FUGA = 7  # "más de 6 palabras" compartidas = fuga


class ErrorValidacion(ValueError):
    """El archivo de evaluación tiene errores. El mensaje los lista todos (no solo el primero)."""


@dataclass
class Consulta:
    id: str
    consulta: str
    doc_id_fuente: str | None
    tipo: str | None
    familia: str | None
    particion: str
    sin_respuesta: bool = False
    borrador: bool = False
    revisado: bool = False     # campo heredado de v11.1; en la evaluación automática cuenta `validada`
    notas: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def documento_conocido(self) -> bool:
        return bool(self.doc_id_fuente) and not self.sin_respuesta


_CAMPOS = ("id", "consulta", "doc_id_fuente", "tipo", "familia", "particion", "sin_respuesta")


def _consulta_desde_dict(d: dict) -> Consulta:
    conocidos = set(_CAMPOS) | {"borrador", "revisado", "notas"}
    return Consulta(
        id=str(d["id"]), consulta=d.get("consulta") or "", doc_id_fuente=d.get("doc_id_fuente") or None,
        tipo=d.get("tipo") or None, familia=d.get("familia") or None, particion=d.get("particion", ""),
        sin_respuesta=bool(d.get("sin_respuesta", False)), borrador=bool(d.get("borrador", False)),
        revisado=bool(d.get("revisado", False)),
        notas=d.get("notas") or "", extra={k: v for k, v in d.items() if k not in conocidos})


def validar_consultas(consultas: list[Consulta], corpus: pd.DataFrame | None = None,
                      familias: dict[str, str] | None = None, permitir_vacias: bool = False) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía si todo está bien).

    Reglas: ids únicos; consulta no vacía (salvo en plantillas); partición válida; una consulta con documento
    fuente debe tener tipo y familia; si se da el corpus, el documento debe existir y su tipo coincidir; si se
    da el mapa de familias, la familia debe coincidir con la del tipo; una consulta sin respuesta no debe
    tener documento fuente.
    """
    problemas = []
    vistos: set[str] = set()
    tipos_doc = corpus.set_index("id")["tipo"].to_dict() if corpus is not None else None
    for c in consultas:
        p = f"[{c.id}]"
        if c.id in vistos:
            problemas.append(f"{p} id duplicado")
        vistos.add(c.id)
        if not c.consulta.strip() and not permitir_vacias:
            problemas.append(f"{p} la consulta está vacía")
        if c.particion not in PARTICIONES:
            problemas.append(f"{p} partición '{c.particion}' no es una de {PARTICIONES}")
        if c.sin_respuesta and c.doc_id_fuente:
            problemas.append(f"{p} una consulta sin respuesta no debe tener doc_id_fuente")
        if c.doc_id_fuente:
            if not c.tipo or not c.familia:
                problemas.append(f"{p} con doc_id_fuente debe tener tipo y familia")
            if tipos_doc is not None:
                if c.doc_id_fuente not in tipos_doc:
                    problemas.append(f"{p} doc_id_fuente '{c.doc_id_fuente}' no está en el corpus")
                elif c.tipo and tipos_doc[c.doc_id_fuente] != c.tipo:
                    problemas.append(f"{p} tipo '{c.tipo}' no coincide con el del corpus "
                                     f"'{tipos_doc[c.doc_id_fuente]}'")
            if familias is not None and c.tipo and c.familia and familias.get(c.tipo, c.familia) != c.familia:
                problemas.append(f"{p} familia '{c.familia}' no coincide con la del mapa ('{familias[c.tipo]}'); "
                                 "¿se revisó familias_penal.csv después de crear la consulta?")
    return problemas


def cargar_consultas(path: Path = config.CONSULTAS_PATH, corpus: pd.DataFrame | None = None,
                     familias: dict[str, str] | None = None, permitir_vacias: bool = False) -> list[Consulta]:
    """Lee y valida un JSONL de consultas. Lanza `ErrorValidacion` con todos los problemas encontrados."""
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Genera una plantilla con "
                                "`python -m copiloto_juridico.evaluacion plantilla` y redacta las consultas.")
    consultas, problemas = [], []
    for n, linea in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip() or linea.lstrip().startswith("#"):  # comentarios (p. ej. el esquema)
            continue
        try:
            d = json.loads(linea)
        except json.JSONDecodeError as e:
            problemas.append(f"línea {n}: JSON inválido ({e})")
            continue
        faltan = [k for k in _CAMPOS if k not in d]
        if faltan:
            problemas.append(f"línea {n}: faltan campos {faltan}")
            continue
        consultas.append(_consulta_desde_dict(d))
    problemas += validar_consultas(consultas, corpus, familias, permitir_vacias)
    if problemas:
        raise ErrorValidacion(f"{path.name}: {len(problemas)} problema(s):\n  - " + "\n  - ".join(problemas))
    return consultas


def cargar_todas(path: Path = config.CONSULTAS_PATH, **kw) -> list[Consulta]:
    """Todas las consultas del conjunto (alias de `cargar_consultas` con la ruta por defecto)."""
    return cargar_consultas(path, **kw)


def guardar_consultas(consultas: list[Consulta], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in consultas:
            d = asdict(c)
            d.update(d.pop("extra"))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def cargar_juicios(path: Path, ids_consultas: set[str] | None = None) -> pd.DataFrame:
    """Lee la hoja de juicios (consulta_id, doc_id, relevancia ∈ {0,1,2}, anotador; puede traer más columnas de
    apoyo, como el resumen) y valida su contenido. Las filas sin relevancia (aún no juzgadas) se ignoran; si falta
    el anotador se asume "anotador1".

    Si un mismo par lo juzgaron varios anotadores, se conservan todas las filas (sirven para calcular kappa);
    `relevancia_por_par` decide qué juicio usar en las métricas.
    """
    vacio = pd.DataFrame(columns=["consulta_id", "doc_id", "relevancia", "anotador"])
    if not path.exists():
        return vacio
    df = pd.read_csv(path, dtype={"consulta_id": str, "doc_id": str, "anotador": str})
    faltan = {"consulta_id", "doc_id", "relevancia"} - set(df.columns)
    if faltan:
        raise ErrorValidacion(f"{path.name}: faltan columnas {sorted(faltan)}")
    df = df[df["relevancia"].notna() & (df["relevancia"].astype(str).str.strip() != "")].copy()
    if df.empty:
        return vacio
    df["anotador"] = df.get("anotador", pd.Series(index=df.index, dtype=str)).fillna("anotador1")
    df["relevancia"] = pd.to_numeric(df["relevancia"], errors="coerce")
    problemas = []
    malas = df[~df["relevancia"].isin([0, 1, 2])]
    for _, r in malas.iterrows():
        problemas.append(f"relevancia '{r['relevancia']}' inválida en ({r['consulta_id']}, {r['doc_id']})")
    if ids_consultas is not None:
        for cid in sorted(set(df["consulta_id"]) - ids_consultas):
            problemas.append(f"consulta_id '{cid}' no existe en las consultas")
    for faceta in FACETAS:  # columnas opcionales: sí / no / vacío
        if faceta in df.columns:
            df[faceta] = df[faceta].map(_si_no)
            for _, r in df[df[faceta].map(lambda v: v == "inválido")].iterrows():
                problemas.append(f"{faceta} debe ser sí/no en ({r['consulta_id']}, {r['doc_id']})")
    dup = df.duplicated(["consulta_id", "doc_id", "anotador"], keep=False)
    if dup.any():
        problemas.append(f"{int(dup.sum())} filas repiten el mismo (consulta_id, doc_id, anotador)")
    if problemas:
        raise ErrorValidacion(f"{path.name}: {len(problemas)} problema(s):\n  - " + "\n  - ".join(problemas))
    df["relevancia"] = df["relevancia"].astype(int)
    return df


FACETAS = ("mismo_delito", "mismo_problema_procesal")


def _si_no(v) -> bool | None | str:
    t = str(v).strip().lower() if v is not None and not (isinstance(v, float) and np.isnan(v)) else ""
    return True if t in ("si", "sí", "s", "1") else False if t in ("no", "n", "0") else None if not t else "inválido"


def facetas_por_par(juicios: pd.DataFrame) -> dict[str, dict[str, dict[str, bool]]]:
    """{consulta_id: {doc_id: {faceta: bool}}} con las facetas que el anotador sí marcó (primer anotador)."""
    out: dict[str, dict[str, dict[str, bool]]] = {}
    for r in juicios.to_dict("records"):
        marcas = {f: r[f] for f in FACETAS if isinstance(r.get(f), bool)}
        if marcas:
            out.setdefault(r["consulta_id"], {}).setdefault(r["doc_id"], marcas)
    return out


def relevancia_por_par(juicios: pd.DataFrame, anotador_principal: str | None = None) -> dict[str, dict[str, int]]:
    """{consulta_id: {doc_id: relevancia}}. Con varios anotadores usa el principal si se indica (el que anotó
    todo); si no, el primero que aparece en el archivo."""
    out: dict[str, dict[str, int]] = {}
    df = juicios
    if anotador_principal:
        df = pd.concat([df[df["anotador"] == anotador_principal], df[df["anotador"] != anotador_principal]])
    for r in df.itertuples():
        out.setdefault(r.consulta_id, {}).setdefault(r.doc_id, int(r.relevancia))
    return out


def exigir_revisadas(consultas: list["Consulta"]) -> None:
    """La partición de prueba solo se usa con consultas revisadas por una persona o, en la evaluación automática
    (v11.2), validadas por `referencias` (comprobaciones de la guía + jueces LLM); nunca con borradores sin validar."""
    sin = [c.id for c in consultas if c.particion == "prueba" and not (c.revisado or c.extra.get("validada"))]
    if sin:
        raise ErrorValidacion(f"Rechazado: {len(sin)} consultas de prueba sin revisar ni validar, p. ej. "
                              f"{', '.join(sin[:5])}. Corre `referencias` (evaluación automática) o revísalas y marca "
                              "revisado: true antes de medir en prueba.")


# --------------------------------------------------------------------------------------------------------------
# Consultas nuevas por familia (`borradores --ampliar`)
# --------------------------------------------------------------------------------------------------------------

def ampliar_plantilla(corpus: pd.DataFrame, familias: dict[str, str], existentes: list[Consulta],
                      n_por_familia: int = 10, min_sentencias_familia: int = 10,
                      semilla: int = config.EVAL_SEED, largo_extracto: int = 1200) -> list[Consulta]:
    """v11.2: entradas NUEVAS para llegar a `n_por_familia` consultas por familia (la mitad en cada partición), con
    sentencias que aún no son fuente de ninguna consulta y tipos aún no cubiertos primero. No toca las existentes;
    los ids siguen la numeración. Sin revisión humana, más consultas dan intervalos de confianza más estrechos."""
    rng = np.random.default_rng(semilla + 1)
    penal = corpus[corpus["materia"] == "penal"].copy()
    penal["familia"] = penal["tipo"].map(familias)
    usados = {c.doc_id_fuente for c in existentes if c.doc_id_fuente}
    hay = Counter((c.familia, c.particion) for c in existentes if c.familia and not c.sin_respuesta)
    tipos_hechos = {c.tipo for c in existentes if c.tipo}
    siguiente = 1 + max([int(c.id[1:]) for c in existentes if re.fullmatch(r"A\d+", c.id)] or [0])
    tam = penal.groupby("familia")["id"].size()
    nuevas = []
    for familia in sorted(tam[tam >= min_sentencias_familia].index):
        grupo = penal[(penal["familia"] == familia) & ~penal["id"].isin(usados)]
        orden = rng.permutation(len(grupo))
        cand = grupo.iloc[orden].assign(_hecho=lambda d: d["tipo"].isin(tipos_hechos)).sort_values("_hecho",
                                                                                                 kind="stable")
        cand = cand.drop_duplicates("tipo").pipe(lambda d: pd.concat([d, cand[~cand["id"].isin(d["id"])]]))
        filas = iter(cand.itertuples())
        for particion in PARTICIONES:
            for _ in range(max(0, n_por_familia // 2 - hay[(familia, particion)])):
                fila = next(filas, None)
                if fila is None:
                    break
                nuevas.append(Consulta(
                    id=f"A{siguiente:03d}", consulta="", doc_id_fuente=fila.id, tipo=fila.tipo, familia=familia,
                    particion=particion, borrador=True, notas="Ampliación v11.2 (evaluación automática).",
                    extra={"origen_muestra": "A-v11.2", "extracto_sentencia": fila.texto[:largo_extracto]}))
                siguiente += 1
    return nuevas


# --------------------------------------------------------------------------------------------------------------
# Fuga de información
# --------------------------------------------------------------------------------------------------------------


def _palabras(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", t)


def ngramas_compartidos(consulta: str, texto_fuente: str, n: int = N_GRAMA_FUGA) -> list[str]:
    """Secuencias de `n` palabras seguidas que aparecen tanto en la consulta como en la sentencia fuente.

    Por qué importa: si la consulta copia frases de la sentencia, BM25 la encuentra trivialmente y la métrica
    mide "copiar y pegar", no la capacidad de encontrar precedentes. Con n = 7 se detecta cualquier coincidencia
    de más de 6 palabras (criterio de la sección 6.2). Se comparan palabras normalizadas (minúsculas, sin
    acentos ni signos), igual que el texto del corpus.
    """
    q = _palabras(consulta)
    if len(q) < n:
        return []
    grams_q = {" ".join(q[i:i + n]) for i in range(len(q) - n + 1)}
    d = _palabras(texto_fuente)
    grams_d = {" ".join(d[i:i + n]) for i in range(len(d) - n + 1)}
    return sorted(grams_q & grams_d)


_MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
_RE_DETALLE = re.compile(rf"\b\d{{1,2}} de (?:{_MESES})(?: de \d{{4}})?|\b(?:19|20)\d{{2}}\b|\b\d{{3,}}\b"
                         r"|\b(?:\d+|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+(?:años?|meses|días)\b",
                         re.IGNORECASE)


def detalles_identificables(consulta: str) -> list[str]:
    """Fechas, años, números largos y duraciones de pena ("28 de junio de 2024", "tres años de prisión"): con ellos
    la consulta apunta a una sola sentencia aunque no copie frases, y la búsqueda se vuelve trivial. Una edad
    ("una menor de 14 años") describe el caso sin identificarlo y no cuenta."""
    salida = []
    for m in _RE_DETALLE.finditer(consulta):
        despues = consulta[m.end():m.end() + 30].lower()
        es_edad = consulta[max(0, m.start() - 3):m.start()].lower() == "de " and "años" in m.group(0).lower()
        if es_edad and not re.search(r"prisi|intern|c[aá]rcel|condena|pena", despues):
            continue
        salida.append(m.group(0))
    return salida


def verificar_fuga(consultas: list[Consulta], corpus: pd.DataFrame, n: int = N_GRAMA_FUGA) -> dict[str, list[str]]:
    """{consulta_id: n-gramas compartidos} para las consultas con fuga (vacío si no hay ninguna)."""
    textos = corpus.set_index("id")["texto"]
    out = {}
    for c in consultas:
        if c.doc_id_fuente and c.consulta and c.doc_id_fuente in textos.index:
            comunes = ngramas_compartidos(c.consulta, textos[c.doc_id_fuente], n)
            if comunes:
                out[c.id] = comunes
    return out


def nombre_tipo_en_consulta(c: Consulta) -> bool:
    """True si la consulta contiene literalmente el nombre del tipo (p. ej. "robo calificado"), otra pista
    que facilita artificialmente la clasificación. Se reporta como advertencia, no como error."""
    if not c.tipo:
        return False
    nombre = " ".join(_palabras(c.tipo.split("/", 1)[-1]))
    return len(nombre.split()) >= 2 and nombre in " ".join(_palabras(c.consulta))
