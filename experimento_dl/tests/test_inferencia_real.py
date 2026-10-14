"""Validación de la inferencia con los checkpoints REALES (criterio de finalización de P1).

Se omite automáticamente mientras falten artifacts/v2/weights/{mil,bigru}_final.pt o no haya
estudios RSNA de validación/prueba en experimento_dl/data/dicom/<UID>/. Cuando existan:
- los checkpoints deben coincidir con el SHA-256 de MANIFEST.json;
- la probabilidad calculada por la app desde DICOM crudo debe reproducir la registrada en la
  evaluación (test/val_predictions_*.csv) con |Δ| ≤ app_core.REPRO_TOL.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app_core as core  # noqa: E402

READY = all(core.checkpoint_status(m, verify_hash=False)["estado"] == "DISPONIBLE" for m in core.MODELS)
EVAL_STUDIES = [u for u in core.local_studies()
                if (core.evaluation_lookup(u) or {}).get("split") in ("val", "test")]

pytestmark = pytest.mark.skipif(not READY, reason="PENDIENTE: faltan los checkpoints reales en artifacts/v2/weights/")


def test_checkpoints_match_manifest():
    for m in core.MODELS:
        assert core.checkpoint_status(m)["estado"] == "DISPONIBLE"


@pytest.mark.skipif(not EVAL_STUDIES, reason="PENDIENTE: no hay estudios RSNA de val/test en experimento_dl/data/dicom/")
def test_app_reproduces_evaluation():
    loaded, skipped = core.load_models(core.MODELS)
    assert not skipped, skipped
    df = core.predict_local_batch(EVAL_STUDIES[:5], loaded)
    assert "error" not in df or df.error.isna().all(), df.get("error")
    assert df.reproduce_evaluacion.all(), df[["StudyInstanceUID", "model", "probabilidad", "prob_evaluacion"]]
