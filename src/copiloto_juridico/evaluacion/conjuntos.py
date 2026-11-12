"""Conjuntos de evaluación: consultas, juicios de relevancia y fichas anotadas (sección 6.2 del informe).

Archivos (en `evaluacion/`):

- `consultas_penal.jsonl`, una consulta por línea:
  `{"id", "consulta", "doc_id_fuente", "tipo", "familia", "particion", "sin_respuesta", "borrador"?, "notas"?}`
  * Conjunto A (documento conocido): `doc_id_fuente` es la sentencia de la que se redactó la consulta.
  * Conjunto B (consultas reales): `doc_id_fuente` es null; su relevancia viene de `juicios.csv`.
  * Sin respuesta: `sin_respuesta: true`, tema sin tipo en el corpus; el sistema debería reconocerlo.
- `juicios.csv`: `consulta_id, doc_id, relevancia (0/1/2), anotador`.
- `fichas_anotadas.jsonl`: `{"doc_id", "sentido_fallo", "etapa", "delito", "articulos": [...], ...}`.

Este módulo **no redacta consultas finales**: `muestrear_plantilla` genera un borrador (`borrador: true`) con
la sentencia elegida y un extracto de sus hechos; una persona redacta la consulta. Los experimentos marcan
como "BORRADOR — no reportable" cualquier corrida que use consultas en borrador.
"""

from __future__ import annotations

import json
import re
import unicodedata
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
    revisado: bool = False     # una persona revisó la consulta (obligatorio para usar la partición de prueba)
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
        if not linea.strip():
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


def guardar_consultas(consultas: list[Consulta], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in consultas:
            d = asdict(c)
            d.update(d.pop("extra"))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def cargar_juicios(path: Path = config.JUICIOS_PATH, ids_consultas: set[str] | None = None) -> pd.DataFrame:
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
    dup = df.duplicated(["consulta_id", "doc_id", "anotador"], keep=False)
    if dup.any():
        problemas.append(f"{int(dup.sum())} filas repiten el mismo (consulta_id, doc_id, anotador)")
    if problemas:
        raise ErrorValidacion(f"{path.name}: {len(problemas)} problema(s):\n  - " + "\n  - ".join(problemas))
    df["relevancia"] = df["relevancia"].astype(int)
    return df


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


def acuerdo_entre_anotadores(juicios: pd.DataFrame) -> dict | None:
    """Kappa de Cohen sobre los pares juzgados por dos anotadores (el 20 % doble del conjunto B)."""
    from .metricas import cohen_kappa

    anotadores = juicios["anotador"].dropna().unique().tolist()
    if len(anotadores) < 2:
        return None
    a, b = anotadores[:2]
    pa = juicios[juicios["anotador"] == a].set_index(["consulta_id", "doc_id"])["relevancia"]
    pb = juicios[juicios["anotador"] == b].set_index(["consulta_id", "doc_id"])["relevancia"]
    comunes = pa.index.intersection(pb.index)
    if len(comunes) == 0:
        return None
    return {"anotadores": [a, b], "pares": int(len(comunes)),
            "kappa_0_1_2": cohen_kappa(pa[comunes].tolist(), pb[comunes].tolist()),
            "kappa_binario": cohen_kappa((pa[comunes] >= 1).tolist(), (pb[comunes] >= 1).tolist())}


def cargar_fichas_anotadas(path: Path = config.FICHAS_ANOTADAS_PATH) -> list[dict]:
    """Conjunto C desde la hoja que llena el abogado: una fila por sentencia con sentido_fallo, etapa,
    acto_impugnado, delito, articulos ("233; 236"), agravios y, opcional, consulta_id + ficha_aceptable (si/no).
    Las filas sin sentido del fallo (aún no anotadas) se ignoran."""
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str).fillna("")
    if "doc_id" not in df.columns:
        raise ErrorValidacion(f"{path.name}: falta la columna doc_id")
    fichas = []
    for fila in df.to_dict("records"):
        if not fila.get("sentido_fallo", "").strip():
            continue
        d = {k: v.strip() for k, v in fila.items() if isinstance(v, str) and v.strip()}
        d["articulos"] = [a.strip() for a in re.split(r"[;,]", fila.get("articulos", "")) if a.strip()]
        if fila.get("ficha_aceptable", "").strip():
            d["ficha_aceptable"] = fila["ficha_aceptable"].strip().lower() in {"si", "sí", "1", "true"}
        fichas.append(d)
    return fichas


def exigir_revisadas(consultas: list["Consulta"]) -> None:
    """La partición de prueba solo se usa con consultas revisadas por una persona (no con borradores del LLM)."""
    sin = [c.id for c in consultas if c.particion == "prueba" and not c.revisado]
    if sin:
        raise ErrorValidacion(f"Rechazado: {len(sin)} consultas de prueba sin revisar (revisado: false), p. ej. "
                              f"{', '.join(sin[:5])}. Revísalas y marca revisado: true antes de medir en prueba.")


# --------------------------------------------------------------------------------------------------------------
# Plantilla de consultas (muestreo estratificado)
# --------------------------------------------------------------------------------------------------------------

TEMAS_SIN_RESPUESTA = [
    "Justicia para adolescentes: medida de internamiento a un menor de 16 años",
    "Delito fiscal federal (defraudación fiscal) — competencia federal, no del STJ",
    "Amparo indirecto contra la orden de aprehensión (lo resuelve un juez federal)",
    "Beneficios preliberacionales en la etapa de ejecución de la pena",
    "Delitos electorales en una jornada de votación",
    "Contrabando de mercancía en aduana",
    "Delitos informáticos: acceso ilícito a sistemas de una empresa",
    "Extradición internacional de un imputado",
    "Responsabilidad penal de una persona moral (empresa)",
    "Delito contra la propiedad industrial (piratería de marcas)",
]


def muestrear_plantilla(corpus: pd.DataFrame, familias: dict[str, str], n_por_familia: int = 3,
                        min_sentencias_familia: int = 10, n_raros: int = 20, n_sin_respuesta: int = 10,
                        semilla: int = config.EVAL_SEED, largo_extracto: int = 1200) -> list[Consulta]:
    """Plantilla del conjunto A con muestreo estratificado por familia y semilla fija.

    - De cada familia con al menos `min_sentencias_familia` sentencias se eligen `n_por_familia` sentencias,
      de tipos distintos cuando es posible (para no medir siempre el tipo más frecuente).
    - Se añaden `n_raros` sentencias de tipos con menos de 5 sentencias (los casos difíciles).
    - Se añaden `n_sin_respuesta` temas sin tipo en el corpus.
    - Partición 50/50 estratificada: dentro de cada familia se alternan desarrollo y prueba tras barajar.

    Cada elemento sale con `borrador: true`, `consulta: ""` y en `extra` un extracto del inicio de la
    sentencia para que la persona que redacta sepa de qué trata. La consulta final la escribe una persona.
    """
    rng = np.random.default_rng(semilla)
    penal = corpus[corpus["materia"] == "penal"].copy()
    penal["familia"] = penal["tipo"].map(familias)
    por_tipo = penal.groupby("tipo")["id"].transform("size")
    tam_familia = penal.groupby("familia")["id"].size()

    elegidos: list[tuple[str, pd.Series]] = []
    usados: set[str] = set()
    for familia in sorted(tam_familia[tam_familia >= min_sentencias_familia].index):
        grupo = penal[penal["familia"] == familia]
        tipos = list(grupo["tipo"].unique())
        rng.shuffle(tipos)
        # Primero un documento de cada tipo distinto (tipos con más sentencias primero), luego al azar.
        tipos.sort(key=lambda t: -int((grupo["tipo"] == t).sum()))
        filas = []
        for t in tipos:
            cand = grupo[(grupo["tipo"] == t) & ~grupo["id"].isin(usados)]
            if len(cand):
                filas.append(cand.iloc[int(rng.integers(len(cand)))])
            if len(filas) >= n_por_familia:
                break
        if len(filas) < n_por_familia:  # familia con pocos tipos: se completa al azar dentro de la familia
            elegidos_ids = {f["id"] for f in filas}
            resto = grupo[~grupo["id"].isin(usados | elegidos_ids)]
            idx = rng.choice(len(resto), size=min(n_por_familia - len(filas), len(resto)), replace=False)
            filas += [resto.iloc[int(i)] for i in sorted(idx)]
        for fila in filas:
            usados.add(fila["id"])
            elegidos.append(("A", fila))

    raros = penal[(por_tipo < 5) & ~penal["id"].isin(usados)]
    if len(raros):
        idx = rng.choice(len(raros), size=min(n_raros, len(raros)), replace=False)
        for i in sorted(idx):
            elegidos.append(("A-raro", raros.iloc[int(i)]))

    # Partición estratificada por familia
    particion: dict[str, str] = {}
    familias_elegidas: dict[str, list[str]] = {}
    for _, fila in elegidos:
        familias_elegidas.setdefault(fila["familia"], []).append(fila["id"])
    inicio = 0
    for fam in sorted(familias_elegidas):
        ids = familias_elegidas[fam]
        rng.shuffle(ids)
        for j, d in enumerate(ids):
            particion[d] = PARTICIONES[(j + inicio) % 2]
        inicio += len(ids)  # alterna con qué partición empieza cada familia para equilibrar las impares

    plantilla = []
    for n, (origen, fila) in enumerate(elegidos, 1):
        plantilla.append(Consulta(
            id=f"A{n:03d}", consulta="", doc_id_fuente=fila["id"], tipo=fila["tipo"], familia=fila["familia"],
            particion=particion[fila["id"]], borrador=True,
            notas="Redactar como lo escribiría un defensor: hechos y problema procesal, sin copiar frases de la "
                  "sentencia ni el nombre del tipo.",
            extra={"origen_muestra": origen, "extracto_sentencia": fila["texto"][:largo_extracto]}))
    for n, tema in enumerate(TEMAS_SIN_RESPUESTA[:n_sin_respuesta], 1):
        plantilla.append(Consulta(
            id=f"S{n:03d}", consulta="", doc_id_fuente=None, tipo=None, familia=None,
            particion=PARTICIONES[n % 2], sin_respuesta=True, borrador=True,
            notas="Tema sugerido sin tipo en el corpus; confirmar que de verdad no hay sentencias del tema.",
            extra={"tema_sugerido": tema}))
    return plantilla


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
