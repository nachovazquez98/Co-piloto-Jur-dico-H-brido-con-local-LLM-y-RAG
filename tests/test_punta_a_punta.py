"""Prueba de punta a punta (bloque 7): pipeline real con datos mínimos, embeddings y LLM simulados. Corre en un
subproceso porque las rutas de datos se fijan al importar `config` (COPILOTO_ROOT)."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_completo_con_llm_simulado(tmp_path):
    env = {**os.environ, "COPILOTO_ROOT": str(tmp_path), "COPILOTO_CHROMA_DIR": str(tmp_path / "chroma"),
           "COPILOTO_EVAL_DIR": str(tmp_path / "evaluacion"), "COPILOTO_FICHAS_PRECALCULADAS": "false",
           "COPILOTO_MATERIAS": "penal", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "e2e" / "correr.py")], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["errores"] == []
    assert len(out["precedentes"]) == 3 and all(t.startswith("penal/robo") for t in out["precedentes"])
    assert set(out["sentidos"]) == {"CONFIRMA"} and set(out["para_defensa"]) == {"desfavorable"}
    assert out["sintesis"].startswith("### Precedentes a favor y en contra")
    assert out["citas"] == {"afirmaciones": 3, "con_cita_valida": 2, "sin_cita": 0, "marcadas": 1}
    assert "_(verificar: la ficha citada dice CONFIRMA)_" in out["sintesis"]
    assert out["llamadas"] == ["conceptos"] + ["qc1", "ficha"] * 3 + ["sintesis"]
