"""Clasificación híbrida numérica (Coseno + Jaccard + KEM + Concisión + Voto) — migrada del notebook.

Las categorías ya no vienen de un caché externo (que no coincidía con la base): se derivan del corpus
indexado, así que la categoría elegida siempre existe como filtro.

El cálculo se separa en dos pasos para poder evaluarlo:
1. `componentes`: para cada categoría candidata, las cinco señales y las dos penalizaciones (no depende de pesos).
2. `puntuar`: combina las señales con unos pesos. La ablación y la búsqueda en malla de pesos reutilizan el
   paso 1 y solo repiten el paso 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .embeddings import embed_texts

logger = logging.getLogger(__name__)

COMPONENTES = ("coseno", "jaccard", "kem", "concision", "voto")
PENALIZACIONES = ("penal_longitud", "penal_tokens")


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    return spacy.load("es_core_news_sm", disable=["parser", "ner"])


def lemmatized_tokens(text: str) -> set[str]:
    """Lemas de las palabras con contenido: minúsculas, sin palabras vacías, signos ni números.

    Lematizar reduce cada palabra a su forma de diccionario ("lesionó", "lesiones" → "lesión") para que la
    comparación con los nombres de las categorías no dependa de la conjugación o el plural.
    """
    doc = get_nlp()(text.lower().replace("-", " ").replace("_", " "))
    return {t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha}


def jaccard(a: set, b: set) -> float:
    """Índice de Jaccard |A∩B| / |A∪B|: palabras en común entre dos conjuntos, relativo a todas las palabras.

    Penaliza los nombres de categoría largos: si la categoría tiene muchas palabras que la consulta no menciona,
    la unión crece y el índice baja.
    """
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def key_entity_match(category_tokens: set, entity_tokens: set, high_importance_tokens: set) -> float:
    """KEM (Key Entity Match): fracción de los términos *importantes* de la consulta que aparecen en la categoría.

    KEM = |R ∩ C| / |R|, con R = términos importantes presentes en la consulta y C = lemas de la categoría.
    A diferencia de Jaccard no castiga que la categoría tenga palabras extra: mide cobertura de la consulta.
    Con R = conceptos de la consulta, KEM queda casi idéntico a Jaccard (mismos conjuntos, distinto denominador):
    por eso su peso es 0 en v11. Se sigue calculando porque la penalización por longitud lo usa.
    """
    relevantes = high_importance_tokens & entity_tokens
    return len(relevantes & category_tokens) / len(relevantes) if relevantes else 0.0


PREFIJO_FAMILIA = "familia/"


def etiqueta_tipo(tipo: str) -> str:
    """`penal/robo-calificado_otros` -> `PENAL · robo calificado / otros`;
    `penal/familia/delitos-sexuales` -> `PENAL · familia: delitos sexuales`."""
    materia, accion = tipo.split("/", 1)
    if accion.startswith(PREFIJO_FAMILIA):
        return f"{materia.upper()} · familia: " + accion[len(PREFIJO_FAMILIA):].replace("-", " ")
    return f"{materia.upper()} · " + accion.replace("_", " / ").replace("-", " ")


@dataclass(frozen=True)
class PesosClasificador:
    """Pesos de las cinco señales y qué penalizaciones se aplican.

    score = Σ peso_i · señal_i, y después se multiplica por las penalizaciones activas. Como solo importa el
    *orden* de las categorías dentro de una consulta, los pesos no necesitan sumar 1 (la búsqueda en malla
    sí los restringe a sumar 1 para que los resultados sean comparables).
    """

    coseno: float = 0.20
    jaccard: float = 0.20
    kem: float = 0.20
    concision: float = 0.05
    voto: float = 0.35
    penal_longitud: bool = True
    penal_tokens: bool = True

    @classmethod
    def desde_config(cls) -> "PesosClasificador":
        return cls(config.COSINE_WEIGHT, config.JACCARD_WEIGHT, config.KEM_WEIGHT, config.CONCISION_WEIGHT,
                   config.VOTE_WEIGHT, True, config.USAR_LISTA_TOKENS_PENALIZADOS)

    def sin(self, componente: str) -> "PesosClasificador":
        """Copia con un componente apagado (peso 0 o penalización desactivada), para la ablación."""
        if componente in PENALIZACIONES:
            return replace(self, **{componente: False})
        if componente in COMPONENTES:
            return replace(self, **{componente: 0.0})
        raise ValueError(f"Componente desconocido: {componente}")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, c) for c in COMPONENTES], dtype=float)


def puntuar(comp: pd.DataFrame, pesos: PesosClasificador) -> np.ndarray:
    """Score híbrido de cada fila de `componentes` con los pesos dados (puntaje para ordenar, no probabilidad)."""
    base = comp[list(COMPONENTES)].to_numpy(dtype=float) @ pesos.vector()
    penal = np.ones(len(comp))
    if pesos.penal_longitud:
        penal *= comp["penal_longitud"].to_numpy(dtype=float)
    if pesos.penal_tokens:
        penal *= comp["penal_tokens"].to_numpy(dtype=float)
    return base * penal


@dataclass
class ResultadoClasificacion:
    tipo: str
    tabla: pd.DataFrame
    tokens_kem: list[str]


class ClasificadorHibrido:
    """Clasificador híbrido sobre un conjunto de categorías.

    A nivel "tipo" cada categoría es un tipo del corpus (`penal/robo-calificado`). A nivel "familia" los tipos
    penales se agrupan en familias de delito (`penal/familia/robo`, ver `evaluacion/familias.py`) y la búsqueda
    con filtro usa todos los tipos de las familias elegidas; los tipos de otras materias quedan como están.
    """

    def __init__(self, tipos: list[str], nivel: str = "tipo", familias: dict[str, str] | None = None,
                 enriquecer: bool | None = None):
        if nivel not in ("tipo", "familia"):
            raise ValueError(f"nivel desconocido: {nivel}")
        self.nivel = nivel
        self.enriquecer = config.ENRIQUECER_FAMILIAS if enriquecer is None else enriquecer
        self.categoria_de: dict[str, str] = {}
        for t in tipos:
            if nivel == "familia" and t.startswith("penal/") and familias is not None:
                self.categoria_de[t] = f"penal/{PREFIJO_FAMILIA}{familias.get(t, t.split('/', 1)[1])}"
            else:
                self.categoria_de[t] = t
        self.miembros: dict[str, list[str]] = {}
        for t, cat in sorted(self.categoria_de.items()):
            self.miembros.setdefault(cat, []).append(t)
        self.tipos = sorted(self.miembros)
        textos = [t.replace(PREFIJO_FAMILIA, "").replace("/", " ").replace("_", " ").replace("-", " ")
                  for t in self.tipos]
        # El embedding de una familia incluye los nombres de sus tipos más generales (los más cortos): "penal
        # contra la salud" solo no dice "narcomenudeo". Jaccard sigue usando solo el nombre de la familia.
        self.embeddings = embed_texts([self._texto_embedding(c, t) for c, t in zip(self.tipos, textos)], kind="query")
        self.tokens = [lemmatized_tokens(t) for t in textos]
        self.indice = {t: i for i, t in enumerate(self.tipos)}

    def _texto_embedding(self, categoria: str, texto: str, max_tipos: int = 6) -> str:
        if PREFIJO_FAMILIA not in categoria or not self.enriquecer:
            return texto
        nombres = sorted({t.split("/", 1)[1].replace("_", " ").replace("-", " ") for t in self.miembros[categoria]},
                         key=len)[:max_tipos]
        return f"{texto}: {', '.join(nombres)}"

    def tipos_de(self, categoria: str) -> list[str]:
        """Tipos del corpus que cubre una categoría (a nivel tipo, la propia categoría)."""
        return self.miembros.get(categoria, [categoria])

    def componentes(self, consulta: str, entidades: str, query_embedding: np.ndarray,
                    recuperados: list | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Señales de cada categoría candidata (las 20 de mayor coseno + las que aparecen en la recuperación).

        - coseno: similitud entre la consulta y el nombre de la categoría, reescalada al rango de los candidatos
          ((cos − mín) / (máx − mín)) porque e5 comprime los cosenos entre 0.75 y 0.90;
        - jaccard y kem: coincidencia de lemas entre los conceptos clave y el nombre de la categoría;
        - concision: 1 / √(número de lemas de la categoría), favorece nombres generales;
        - voto: suma del RRF de las sentencias recuperadas de esa categoría, dividida entre el máximo;
        - penal_longitud: 0.90 si KEM < 0.6 y la categoría tiene más de 5 lemas, si no 1;
        - penal_tokens: 0.70 si la categoría contiene una palabra de `config.IRRELEVANT_TOKENS`, si no 1.
        """
        entity_tokens = lemmatized_tokens(entidades) if entidades else lemmatized_tokens(consulta)
        high_importance = entity_tokens  # KEM = |A∩B|/|A| con A = conceptos (peso 0 en v11, ver config)

        cos = self.embeddings @ query_embedding
        votos: dict[str, float] = {}
        for c in recuperados or []:
            cat = self.categoria_de.get(c.tipo, c.tipo)
            votos[cat] = votos.get(cat, 0.0) + c.relevancia
        max_voto = max(votos.values(), default=0.0) or 1.0

        candidatos = list(np.argsort(cos)[-config.TOP_N_CANDIDATES_INITIAL:][::-1])
        candidatos += [self.indice[t] for t in votos if t in self.indice and self.indice[t] not in candidatos]
        cos_cand = cos[candidatos]
        cos_min, cos_rng = cos_cand.min(), max(1e-6, cos_cand.max() - cos_cand.min())

        filas = []
        for i in candidatos:
            cat_tokens = self.tokens[i]
            kem = key_entity_match(cat_tokens, entity_tokens, high_importance)
            num_tokens = max(1, len(cat_tokens))
            filas.append({
                "tipo": self.tipos[i], "coseno_bruto": float(cos[i]),
                "coseno": float((cos[i] - cos_min) / cos_rng),
                "jaccard": jaccard(entity_tokens, cat_tokens), "kem": kem,
                "concision": 1 / np.sqrt(num_tokens), "voto": votos.get(self.tipos[i], 0.0) / max_voto,
                "tokens": num_tokens,
                "penal_longitud": 0.90 if kem < config.KEM_THRESHOLD_FOR_PENALTY
                and num_tokens > config.MAX_TOKENS_FOR_PENALTY else 1.0,
                "penal_tokens": config.IRRELEVANT_TOKENS_PENALTY_FACTOR
                if config.IRRELEVANT_TOKENS & cat_tokens else 1.0,
            })
        return pd.DataFrame(filas), sorted(high_importance)

    @staticmethod
    def tabla_desde_componentes(comp: pd.DataFrame, pesos: PesosClasificador,
                                top: int = config.TOP_N_CANDIDATES_FINAL) -> pd.DataFrame:
        """Tabla ordenada por score híbrido con las columnas que muestra la interfaz."""
        t = comp.assign(score=puntuar(comp, pesos))
        t["penalidad"] = (t["penal_longitud"] if pesos.penal_longitud else 1.0) * \
                         (t["penal_tokens"] if pesos.penal_tokens else 1.0)
        t = t.sort_values("score", ascending=False, kind="mergesort").head(top).reset_index(drop=True)
        return pd.DataFrame({
            "tipo": t["tipo"], "categoría": t["tipo"].map(etiqueta_tipo), "score_híbrido": t["score"],
            "coseno": t["coseno_bruto"], "jaccard": t["jaccard"], "kem": t["kem"],
            "voto_recuperación": t["voto"], "concisión": t["concision"], "tokens": t["tokens"],
            "penalidad": t["penalidad"]})

    def clasificar(self, consulta: str, entidades: str, query_embedding: np.ndarray,
                   recuperados: list | None = None, pesos: PesosClasificador | None = None) -> ResultadoClasificacion:
        """Re-ranking híbrido. Además de las métricas del notebook, suma un *voto de recuperación*: qué tipos
        aparecen entre las sentencias más relevantes de la búsqueda híbrida (ponderado por su RRF)."""
        comp, tokens = self.componentes(consulta, entidades, query_embedding, recuperados)
        tabla = self.tabla_desde_componentes(comp, pesos or PesosClasificador.desde_config())
        return ResultadoClasificacion(tipo=tabla.iloc[0]["tipo"], tabla=tabla, tokens_kem=tokens)


def construir_clasificador(tipos: list[str], nivel: str | None = None) -> ClasificadorHibrido:
    """Clasificador sobre los tipos del corpus según el alcance configurado: solo las materias de
    `config.MATERIAS_ACTIVAS` y al nivel de `config.NIVEL_CLASIFICACION`."""
    activos = [t for t in tipos if t.split("/", 1)[0] in config.MATERIAS_ACTIVAS] or tipos
    nivel = nivel or config.NIVEL_CLASIFICACION
    familias = None
    if nivel == "familia":
        from .evaluacion.familias import cargar_familias

        familias = cargar_familias(activos)
    return ClasificadorHibrido(activos, nivel=nivel, familias=familias)


def senales_coincidencia(tabla: pd.DataFrame, n_filtro: int = config.NUM_CATEGORIAS_FILTRO) -> dict[str, float]:
    """Señales de qué tan bien coincide la mejor categoría con la consulta, para el aviso "no hay precedentes
    del mismo tipo": score híbrido de la primera, coseno de la primera y KEM máximo entre las del filtro."""
    if tabla.empty:
        return {"score_hibrido": 0.0, "coseno": 0.0, "kem_max": 0.0}
    return {"score_hibrido": float(tabla.iloc[0]["score_híbrido"]), "coseno": float(tabla.iloc[0]["coseno"]),
            "kem_max": float(tabla["kem"].head(n_filtro).max())}
