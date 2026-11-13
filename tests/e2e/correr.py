"""Punta a punta con datos mínimos (lo ejecuta tests/test_punta_a_punta.py en un subproceso, con COPILOTO_ROOT en una
carpeta temporal): corpus -> índice -> Chroma + BM25 -> clasificador -> pipeline completo. Lo único simulado es el
modelo de embeddings (bolsa de palabras con hash, determinista) y el LLM (respuestas fijas). Imprime un JSON."""

import json
import re
import zlib

import numpy as np
import pandas as pd

from copiloto_juridico import config, embeddings


class EmbeddingsFalsos:
    def encode(self, textos, **kw):
        out = np.zeros((len(textos), 64), dtype=np.float32)
        for i, t in enumerate(textos):
            for w in re.findall(r"[a-záéíóúñ]{4,}", t.lower()):
                out[i, zlib.crc32(w.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


embeddings.get_model = lambda *a, **k: EmbeddingsFalsos()

FICHA = {"tribunal": "Sala Penal", "delitos": "Robo calificado", "apelante": "La defensa del imputado",
         "acto_impugnado": "Auto de vinculación a proceso", "sentido_fallo": "CONFIRMA",
         "resumen": "La defensa apeló la vinculación por robo; la Sala la confirmó.",
         "agravios": [{"agravio": "El reconocimiento fue ilegal", "calificacion": "INFUNDADO", "razon": "Hubo defensor"}],
         "fundamento_legal": ["artículo 233"]}


class LLMSimulado:
    settings = config.LLMSettings(provider="openai", model="simulado")

    def __init__(self):
        self.llamadas = []

    def generate(self, prompt, json_mode=False):
        if json_mode:
            self.llamadas.append("ficha")
            return json.dumps(FICHA, ensure_ascii=False)
        if "conceptos jurídicos clave" in prompt:
            self.llamadas.append("conceptos")
            return "robo con violencia, vinculación a proceso"
        self.llamadas.append("qc1")
        return "1. Mismo delito.\n2. Mismo problema procesal.\nVEREDICTO: SI"

    def stream(self, prompt, json_mode=False):
        self.llamadas.append("sintesis")
        yield ("### Respuesta breve\nLa Sala confirmó la vinculación por robo pese al reconocimiento [P1]. "
               "También confirmó la vinculación en el segundo caso de robo [P2].\n"
               "- En el tercer precedente la Sala revocó el auto de robo [P2].")


def sentencia(i, delito, acto):
    return (f"vistos para resolver el toca penal {100 + i}/2025, formado con motivo del recurso de apelación interpuesto "
            f"por la defensa del imputado, en contra del auto de {acto} por el delito de {delito} con violencia en la "
            f"vía pública. " + "la defensa alega que el reconocimiento del imputado fue ilegal. " * 30
            + "resolutivos: primero. se confirma el auto de vinculación a proceso. notifíquese. art. 233.")


filas = [{"id": f"2025-01-0{i}-10000{i}_penal_{tipo}.pdf", "fecha": f"2025-01-0{i}", "materia": "penal",
          "delito_o_accion": tipo, "tipo": f"penal/{tipo}", "texto": sentencia(i, delito, acto), "origen": "prueba"}
         for i, (tipo, delito, acto) in enumerate([
             ("robo-calificado", "robo calificado", "vinculación a proceso"),
             ("robo-calificado", "robo calificado", "vinculación a proceso"),
             ("robo-a-negocio", "robo a negocio", "vinculación a proceso"),
             ("homicidio-calificado", "homicidio calificado", "prisión preventiva"),
             ("homicidio-calificado", "homicidio calificado", "prisión preventiva"),
             ("fraude-generico", "fraude", "no vinculación a proceso")], 1)]

from copiloto_juridico import metadatos                         # noqa: E402
from copiloto_juridico.clasificador import construir_clasificador  # noqa: E402
from copiloto_juridico.datos.corpus import load_corpus, save_corpus  # noqa: E402
from copiloto_juridico.datos.indice import build_index          # noqa: E402
from copiloto_juridico.pipeline import Copiloto                  # noqa: E402
from copiloto_juridico.vectorstore import SentenciasStore        # noqa: E402

save_corpus(pd.DataFrame(filas))
build_index(load_corpus(), log=lambda m: None)
metadatos.guardar(metadatos.calcular(load_corpus()))
store = SentenciasStore(log=lambda m: None)
llm = LLMSimulado()
cop = Copiloto(store, construir_clasificador(store.corpus["tipo"].unique().tolist()), llm)
eventos = list(cop.analizar("A mi cliente lo vincularon a proceso por robo con violencia de un celular en la calle; "
                            "lo reconocieron sin defensor."))
res = next(e.datos["resultado"] for e in eventos if e.tipo == "resultado")
print(json.dumps({
    "errores": [e.mensaje for e in eventos if e.tipo == "error"],
    "categoria": res.categoria, "precedentes": [p.candidato.tipo for p in res.precedentes],
    "sentidos": [p.tarjeta()["sentido_fallo"] for p in res.precedentes],
    "para_defensa": [p.tarjeta()["para_defensa"] for p in res.precedentes],
    "sintesis": res.sintesis, "citas": res.citas, "llamadas": llm.llamadas}, ensure_ascii=False))
