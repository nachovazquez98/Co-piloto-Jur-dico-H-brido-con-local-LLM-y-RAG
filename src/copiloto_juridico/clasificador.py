"""Clasificación híbrida numérica (Coseno + Jaccard + KEM + Concisión) — migrada del notebook.

Las categorías ya no vienen de un caché externo (que no coincidía con la base): se derivan del corpus
indexado, así que la categoría elegida siempre existe como filtro.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .embeddings import embed_texts

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    return spacy.load("es_core_news_sm", disable=["parser", "ner"])


def lemmatized_tokens(text: str) -> set[str]:
    doc = get_nlp()(text.lower().replace("-", " ").replace("_", " "))
    return {t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def key_entity_match(category_tokens: set, entity_tokens: set, high_importance_tokens: set) -> float:
    relevantes = high_importance_tokens & entity_tokens
    return len(relevantes & category_tokens) / len(relevantes) if relevantes else 0.0


def etiqueta_tipo(tipo: str) -> str:
    """`penal/robo-calificado_otros` -> `PENAL · robo calificado / otros`"""
    materia, accion = tipo.split("/", 1)
    return f"{materia.upper()} · " + accion.replace("_", " / ").replace("-", " ")


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

    def clasificar(self, consulta: str, entidades: str, query_embedding: np.ndarray,
                   recuperados: list | None = None) -> ResultadoClasificacion:
        """Re-ranking híbrido. Además de las métricas del notebook, suma un *voto de recuperación*: qué tipos
        aparecen entre las sentencias más relevantes de la búsqueda híbrida (ponderado por su RRF)."""
        entity_tokens = lemmatized_tokens(entidades) if entidades else lemmatized_tokens(consulta)
        high_importance = entity_tokens

        cos = self.embeddings @ query_embedding
        votos: dict[str, float] = {}
        for c in recuperados or []:
            votos[c.tipo] = votos.get(c.tipo, 0.0) + c.relevancia
        max_voto = max(votos.values(), default=0.0) or 1.0

        indice = {t: i for i, t in enumerate(self.tipos)}
        candidatos = list(np.argsort(cos)[-config.TOP_N_CANDIDATES_INITIAL:][::-1])
        candidatos += [indice[t] for t in votos if t in indice and indice[t] not in candidatos]
        cos_cand = cos[candidatos]
        cos_min, cos_rng = cos_cand.min(), max(1e-6, cos_cand.max() - cos_cand.min())

        filas = []
        for i in candidatos:
            cat_tokens = self.tokens[i]
            cos_norm = (cos[i] - cos_min) / cos_rng  # e5 comprime los cosenos (0.75-0.90): se reescalan
            jac = jaccard(entity_tokens, cat_tokens)
            kem = key_entity_match(cat_tokens, entity_tokens, high_importance)
            voto = votos.get(self.tipos[i], 0.0) / max_voto
            num_tokens = max(1, len(cat_tokens))
            concision = 1 / np.sqrt(num_tokens)
            penal = 1.0
            if kem < config.KEM_THRESHOLD_FOR_PENALTY and num_tokens > config.MAX_TOKENS_FOR_PENALTY:
                penal *= 0.90
            if config.IRRELEVANT_TOKENS & cat_tokens:
                penal *= config.IRRELEVANT_TOKENS_PENALTY_FACTOR
            base = (config.COSINE_WEIGHT * cos_norm + config.JACCARD_WEIGHT * jac + config.KEM_WEIGHT * kem
                    + config.CONCISION_WEIGHT * concision + config.VOTE_WEIGHT * voto)
            filas.append({"tipo": self.tipos[i], "categoría": etiqueta_tipo(self.tipos[i]),
                          "score_híbrido": base * penal, "coseno": float(cos[i]), "jaccard": jac, "kem": kem,
                          "voto_recuperación": voto, "concisión": concision, "tokens": num_tokens,
                          "penalidad": penal})
        tabla = pd.DataFrame(filas).sort_values("score_híbrido", ascending=False).head(config.TOP_N_CANDIDATES_FINAL)
        tabla = tabla.reset_index(drop=True)
        return ResultadoClasificacion(tipo=tabla.iloc[0]["tipo"], tabla=tabla, tokens_kem=sorted(entity_tokens))
