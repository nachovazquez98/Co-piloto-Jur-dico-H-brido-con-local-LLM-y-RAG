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
    Nota: si los "términos importantes" son todos los conceptos de la consulta (modo por defecto de v10),
    R = conceptos y KEM queda muy correlacionado con Jaccard (mismos conjuntos, distinto denominador).
    """
    relevantes = high_importance_tokens & entity_tokens
    return len(relevantes & category_tokens) / len(relevantes) if relevantes else 0.0


def etiqueta_tipo(tipo: str) -> str:
    """`penal/robo-calificado_otros` -> `PENAL · robo calificado / otros`"""
    materia, accion = tipo.split("/", 1)
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
                   config.VOTE_WEIGHT, True, True)

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
    def __init__(self, tipos: list[str]):
        self.tipos = sorted(tipos)
        textos = [t.replace("/", " ").replace("_", " ").replace("-", " ") for t in self.tipos]
        self.embeddings = embed_texts(textos, kind="query")
        self.tokens = [lemmatized_tokens(t) for t in textos]
        self.indice = {t: i for i, t in enumerate(self.tipos)}

    def tipos_de(self, categoria: str) -> list[str]:
        """Tipos del corpus que cubre una categoría (a nivel tipo, la propia categoría)."""
        return [categoria]

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
        high_importance = entity_tokens

        cos = self.embeddings @ query_embedding
        votos: dict[str, float] = {}
        for c in recuperados or []:
            votos[c.tipo] = votos.get(c.tipo, 0.0) + c.relevancia
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
        return pd.DataFrame(filas), sorted(entity_tokens)

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


def construir_clasificador(tipos: list[str]) -> ClasificadorHibrido:
    """Clasificador sobre los tipos del corpus, según el alcance configurado (`config.MATERIAS_ACTIVAS`)."""
    activos = [t for t in tipos if t.split("/", 1)[0] in config.MATERIAS_ACTIVAS]
    return ClasificadorHibrido(activos or tipos)
