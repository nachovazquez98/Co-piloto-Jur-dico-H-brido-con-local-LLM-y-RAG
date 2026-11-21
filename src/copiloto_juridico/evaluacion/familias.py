"""Agrupación de los tipos penales en *familias de delito*.

Por qué existe: el portal del STJ etiqueta cada sentencia con un tipo muy fino (289 tipos penales; 139 con una
sola sentencia). Un clasificador no puede aprender ni evaluarse en una clase con un ejemplo, y para el defensor
"robo calificado" y "robo calificado en grado de tentativa" son el mismo problema. La familia agrupa tipos que
un abogado consideraría el mismo delito base.

Cómo se construye: reglas sobre el *slug* del tipo (`penal/robo-calificado-y-otro`). El delito que aparece
primero en el nombre se toma como el principal; si el nombre menciona delitos de otras familias, el tipo se
marca `revisar`. `exportar_csv` escribe la tabla completa en `evaluacion/familias_penal.csv` (todos los tipos penales
del corpus, con la regla que decidió cada familia); si una fila se corrige a mano, `cargar_familias` usa el archivo.
Ninguna asignación ha sido revisada por un especialista: es una limitación declarada.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import config

# Prefijos de grado de ejecución o de continuidad que no cambian el delito base.
_PREFIJOS_TENTATIVA = re.compile(r"^(tentativa|tentantiva|continuado)-de-")

# Reglas ordenadas: (familia, expresión aplicada al inicio del slug, nota para el revisor o "").
# El orden importa: "violacion-de-deposito" debe evaluarse antes que "violacion", y "robo-de-infante" antes que "robo".
REGLAS: list[tuple[str, str, str]] = [
    # v11.2: asuntos sin delito (recursos e incidencias del procedimiento); antes caían en "otros".
    ("cuestiones-procesales", r"queja|conflicto-de-competencia|excusa",
     "No es un delito: recurso o incidencia del procedimiento."),
    ("sustraccion-de-menores", r"robo-de-infante|substraccion-de-menores|sustraccion-de-menores|sustraccion-robo",
     "Robo de infante agrupado con sustracción de menores (delito contra las personas, no patrimonial)."),
    ("robo", r"robo|asalto|abigeato", ""),
    ("feminicidio", r"feminicidio",
     "Se separa de homicidio por su tratamiento propio; confirmar si debe ir dentro de homicidio."),
    ("homicidio", r"homicidio|parricidio|instigacion-o-ayuda-al-suicidio", ""),
    ("lesiones", r"lesiones", ""),
    ("fraude-y-abuso-de-confianza", r"violacion-de-deposito",
     "Violación de depósito es un delito patrimonial, no sexual."),
    ("delitos-sexuales",
     r"violacion|abuso-sexual|acoso-sexual|hostigamiento|aprovechamiento-sexual|atentados-al-pudor|lenocinio",
     ""),
    ("delitos-sexuales", r"prostitucion",
     "Prostitución infantil y explotación sexual agrupadas con delitos sexuales, como el lenocinio."),
    ("delitos-sexuales", r"corrupcion-de-menores",
     "Corrupción de menores agrupada con delitos sexuales; confirmar."),
    ("violencia-familiar", r"violencia-familiar|violencia-intrafamiliar", ""),
    ("violencia-familiar", r"maltrato", "Maltrato infantil agrupado con violencia familiar; confirmar."),
    ("fraude-y-abuso-de-confianza", r"fraude-procesal",
     "Fraude procesal afecta a la administración de justicia; se agrupa con fraude por afinidad. Confirmar."),
    ("fraude-y-abuso-de-confianza", r"fraude|administracion-fraudulenta|abuso-de-confianza", ""),
    ("despojo", r"despojo", ""),
    ("dano-en-las-cosas", r"danos?-(en|a)-las-cosas", ""),
    ("privacion-de-la-libertad-y-secuestro", r"secuestro|plagio|privacion-ilegal|delitos-en-materia-de-secuestro", ""),
    ("desaparicion-de-personas", r"desaparicion", ""),
    ("contra-la-salud", r"contra-la-salud|narcomenudeo", ""),
    ("abandono-de-familiares", r"abandono", ""),
    ("amenazas-y-extorsion", r"amenazas|amenzas|chantaje|extorsion",
     ""),
    ("servidores-publicos-y-administracion-de-justicia",
     r"abuso-de-autoridad|cohecho|peculado|uso-ilicito-de-atribuciones|desvio-y-aprovechamiento"
     r"|delitos-cometidos-en-la-administracion|cometidos-en-la-administracion|delitos-en-materia-de-seguridad-social"
     r"|tortura|abogados-patronos|materia-de-informacion-publica|utilizacion-ilicita-de-informacion|encubrimiento"
     r"|ejercicio-indebido",
     ""),
    ("falsedad-y-falsificacion", r"falsificacion|falsedad|suplantacion-de-identidad|uso-de-documento-falso", ""),
    ("armas", r"portacion-de-arma|armas-y-objetos|disparo-de-arma", ""),
    ("delincuencia-organizada", r"delincuencia-organizada|asociacion-delictuosa", ""),
    ("contra-la-dignidad-de-las-personas", r"contra-la-dignidad", ""),
    ("trata-de-personas", r"trata-de-personas", ""),
    ("responsabilidad-profesional", r"responsabilidad-medica|responsabilidad-profesional", ""),
]

# Los delitos a título de culpa (accidentes, sobre todo de tránsito) se agrupan aparte del doloso equivalente.
FAMILIA_CULPOSOS = "delitos-culposos"
_RE_CULPA = re.compile(r"culp")
NOTA_CULPOSOS = ("Delitos a título de culpa (homicidio, lesiones o daño culposos, típicamente de tránsito) se "
                 "separan de los dolosos. Confirmar.")
FAMILIA_OTROS = "otros"

_REGLAS_COMPILADAS = [(f, re.compile(rf"^(?:{p})"), nota) for f, p, nota in REGLAS]
_REGLAS_BUSQUEDA = [(f, re.compile(rf"(?:^|-)(?:{p})")) for f, p, _ in REGLAS]


@dataclass(frozen=True)
class AsignacionFamilia:
    tipo: str
    familia: str
    regla: str        # qué regla decidió la familia
    revisar: bool     # True si conviene confirmarla (tipo compuesto o sin regla)
    nota: str


def _accion(tipo: str) -> str:
    return tipo.split("/", 1)[1] if "/" in tipo else tipo


def asignar_familia(tipo: str) -> AsignacionFamilia:
    """Familia de un tipo penal según las reglas. Los tipos de otras materias son su propia familia.

    Pasos: (1) si el delito es a título de culpa, familia de culposos; (2) se quita el prefijo de tentativa;
    (3) la primera regla cuyo patrón coincide con el *inicio* del slug decide; (4) si el resto del nombre
    menciona delitos de otras familias ("homicidio-calificado-y-robo-calificado"), se marca `revisar`.
    """
    if not tipo.startswith("penal/"):
        return AsignacionFamilia(tipo, tipo, "no penal", False, "")
    accion = _accion(tipo)
    base = _PREFIJOS_TENTATIVA.sub("", accion)

    resto = base
    if _RE_CULPA.search(accion):
        familia, regla, nota = FAMILIA_CULPOSOS, "culpa", NOTA_CULPOSOS
    else:
        familia, regla, nota = FAMILIA_OTROS, "ninguna", "Ninguna regla coincide."
        for f, patron, n in _REGLAS_COMPILADAS:
            coincidencia = patron.match(base)
            if coincidencia:
                familia, regla, nota = f, patron.pattern, n
                resto = base[coincidencia.end():]  # lo que sigue al delito principal
                break

    otras = sorted({f for f, p in _REGLAS_BUSQUEDA if p.search(resto)} - {familia})
    if familia == FAMILIA_CULPOSOS:
        otras = []  # en los culposos la mezcla homicidio/lesiones/daño es lo normal
    if otras:
        nota = (nota + " " if nota else "") + f"Tipo compuesto: menciona también {', '.join(otras)}."
    revisar = bool(nota) or familia == FAMILIA_OTROS
    return AsignacionFamilia(tipo, familia, regla, revisar, nota.strip())


def etiqueta_familia(familia: str) -> str:
    """`fraude-y-abuso-de-confianza` -> `fraude y abuso de confianza`."""
    return _accion(familia).replace("-", " ")


def asignar_todas(tipos: list[str]) -> list[AsignacionFamilia]:
    return [asignar_familia(t) for t in sorted(set(tipos))]


def exportar_csv(tipos_con_conteo: dict[str, int], path: Path = config.FAMILIAS_CSV,
                 forzar: bool = False) -> Path:
    """Escribe la tabla tipo → familia.

    Columnas: tipo, sentencias, familia (editable), regla, revisar, nota, revisado ("si" si una persona la corrigió).
    No sobrescribe un archivo existente salvo con `forzar`, para no perder una revisión.
    """
    if path.exists() and not forzar:
        raise FileExistsError(f"{path} ya existe (puede contener correcciones a mano). Usa forzar=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "sentencias", "familia", "regla", "revisar", "nota", "revisado"])
        for a in asignar_todas(list(tipos_con_conteo)):
            w.writerow([a.tipo, tipos_con_conteo[a.tipo], a.familia, a.regla, "si" if a.revisar else "no",
                        a.nota, "no"])
    return path


@lru_cache(maxsize=4)
def _leer_csv(path_str: str, mtime: float) -> dict[str, str]:
    with open(path_str, encoding="utf-8") as f:
        return {fila["tipo"].strip(): fila["familia"].strip() for fila in csv.DictReader(f)
                if fila.get("tipo") and fila.get("familia")}


def cargar_familias(tipos: list[str], path: Path = config.FAMILIAS_CSV) -> dict[str, str]:
    """Mapa tipo → familia. Usa el CSV si existe; los tipos que no estén en el CSV
    (p. ej. nuevos tras ampliar el corpus) se asignan con las reglas."""
    revisado = _leer_csv(str(path), path.stat().st_mtime) if path.exists() else {}
    return {t: revisado.get(t) or asignar_familia(t).familia for t in tipos}



def coherencia(doc_ids, emb, doc_tipo: dict[str, str], familias: dict[str, str], min_docs: int = 3):
    """Comprobación sin LLM de la agrupación (v11.2): para cada tipo con al menos
    `min_docs` sentencias, a qué familia se parece más su centroide de embeddings (el promedio de los fragmentos de
    sus sentencias, centrado). La familia de comparación se calcula SIN ese tipo. Un tipo más cercano a otra familia que a la
    suya es un posible error de agrupación (o un tipo con varios delitos): diagnóstico, no una decisión."""
    import numpy as np
    import pandas as pd

    doc_ids = np.asarray(doc_ids)
    e = np.asarray(emb, dtype=np.float32)
    e = e - e.mean(axis=0)  # sin el componente común (los embeddings de e5 tienen todos coseno ~0.99 entre sí)
    df = pd.DataFrame({"doc": doc_ids, "i": np.arange(len(doc_ids))})
    df = df[df["doc"].map(lambda d: d in doc_tipo)]
    vec_doc = {d: e[g["i"].to_numpy()].mean(axis=0) for d, g in df.groupby("doc")}
    por_tipo: dict[str, list] = {}
    for d, v in vec_doc.items():
        por_tipo.setdefault(doc_tipo[d], []).append(v)
    suma = {t: np.sum(vs, axis=0) for t, vs in por_tipo.items()}
    n = {t: len(vs) for t, vs in por_tipo.items()}
    fam_suma: dict[str, np.ndarray] = {}
    fam_n: dict[str, int] = {}
    for t in suma:
        f = familias[t]
        fam_suma[f] = fam_suma.get(f, 0) + suma[t]
        fam_n[f] = fam_n.get(f, 0) + n[t]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    filas = []
    for t in sorted(suma):
        if n[t] < min_docs:
            continue
        c = suma[t] / n[t]
        sims = {}
        for f in fam_suma:
            s, k = (fam_suma[f] - suma[t], fam_n[f] - n[t]) if f == familias[t] else (fam_suma[f], fam_n[f])
            if k > 0:
                sims[f] = cos(c, s / k)
        if familias[t] not in sims:
            continue  # el tipo es toda su familia: no hay con qué comparar
        cercana = max(sims, key=sims.get)
        filas.append({"tipo": t, "sentencias": n[t], "familia asignada": familias[t], "familia más cercana": cercana,
                      "similitud asignada": sims[familias[t]], "similitud más cercana": sims[cercana],
                      "coincide": cercana == familias[t]})
    return pd.DataFrame(filas)
