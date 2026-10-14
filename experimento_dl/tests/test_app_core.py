"""Pruebas funcionales de la app (lógica en app_core + arranque de app_streamlit).

Ejecutar desde la raíz:  .venv/bin/python -m pytest experimento_dl/tests -q

Los DICOM de prueba se derivan de CT_small.dcm (un corte de TC real incluido en pydicom),
replicado con distintas posiciones z para simular un estudio: SOLO verifica la mecánica del
código. El checkpoint de prueba tiene pesos aleatorios y vive en un directorio temporal; sus
salidas no son resultados experimentales.
"""
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_core as core  # noqa: E402


# ----------------------------------------------------------------------------- utilidades
def _ct_slice():
    import pydicom
    from pydicom.data import get_testdata_file
    return pydicom.dcmread(get_testdata_file("CT_small.dcm"))


def fake_study(n, uid="1.2.3.4"):
    base = _ct_slice()
    out = []
    for i in range(n):
        d = copy.deepcopy(base)
        d.StudyInstanceUID = uid
        d.InstanceNumber = i + 1
        d.ImagePositionPatient = [0.0, 0.0, float(i) * 1.0]
        out.append(d)
    return out


def to_bytes(ds):
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


class Upload:  # imita st.UploadedFile
    def __init__(self, name, data):
        self.name, self._d = name, data

    def getvalue(self):
        return self._d


# ----------------------------------------------------------------------------- artefactos
def test_v2_configuration_is_active():
    import common
    assert common.EXP == "v2" and common.K_SLICES == 48 and common.CONTEXT_MM == 2.0
    assert core.preprocessing_config()["k_slices"] == common.K_SLICES


def test_real_metrics_load_and_match_comparison_csv():
    comp = core.comparison_table()
    for m in core.MODELS:
        mt = core.metrics(m)
        assert mt is not None, f"falta metrics_{m}.json"
        row = comp[(comp.modelo.str.contains("MIL" if m == "mil" else "BiGRU"))
                   & (comp.conjunto == "prueba (umbral val)")].iloc[0]
        assert abs(row.roc_auc - mt["test_thr_val_youden"]["roc_auc"]) < 1e-3


def test_metrics_at_threshold_reproduces_published_confusion_matrix():
    for m in core.MODELS:
        pr, mt = core.predictions(m), core.metrics(m)["test_thr_val_youden"]
        r = core.metrics_at_threshold(pr.patient_overall, pr.prob, mt["threshold"])
        assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (mt["tp"], mt["fp"], mt["tn"], mt["fn"])
        _, auc = core.roc_points(pr.patient_overall, pr.prob)
        assert abs(auc - mt["roc_auc"]) < 1e-6


def test_mil_slice_probs_load_after_hash_check():
    uids, probs = core.mil_test_slice_probs()
    assert len(uids) == 404 and probs.shape == (404, 48)


def test_metrics_long_shape():
    long = core.metrics_long()
    assert len(long) == len(core.MODELS) * len(core.SETS) * len(core.METRIC_KEYS)
    assert long.valor.between(0, 1).all()


def test_missing_checkpoint_is_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "WEIGHTS_DIR", tmp_path)
    for m in core.MODELS:
        assert core.checkpoint_status(m)["estado"] == "PENDIENTE"
    loaded, skipped = core.load_models(core.MODELS)
    assert loaded == {} and set(skipped) == set(core.MODELS)


def test_wrong_checkpoint_is_rejected_by_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "WEIGHTS_DIR", tmp_path)
    (tmp_path / "bigru_final.pt").write_bytes(b"no es el checkpoint publicado")
    assert core.checkpoint_status("bigru")["estado"] == "HASH DISTINTO"
    assert core.load_models(["bigru"])[0] == {}


# ----------------------------------------------------------------------------- entradas inválidas
def test_empty_input_raises():
    with pytest.raises(core.InvalidInput):
        core.read_dicoms([])


def test_non_dicom_files_raise():
    with pytest.raises(core.InvalidInput):
        core.read_dicoms([Upload("a.dcm", b"esto no es DICOM"), Upload("b.dcm", b"")])


def test_too_many_files_raise():
    with pytest.raises(core.InvalidInput):
        core.read_dicoms([Upload("x.dcm", b"x")] * 3, max_files=2)


def test_mixed_valid_and_invalid_files():
    good = [Upload(f"{i}.dcm", to_bytes(d)) for i, d in enumerate(fake_study(3))]
    dsets, errors, warns = core.read_dicoms(good + [Upload("malo.dcm", b"basura")])
    assert len(dsets) == 3 and len(errors) == 1 and "malo.dcm" in errors[0]
    assert any("< 48" in w for w in warns)


def test_mixed_studies_warns():
    a, b = fake_study(2, "1.2.3.5"), fake_study(2, "1.2.3.6")
    _, _, warns = core.read_dicoms([Upload(f"{i}.dcm", to_bytes(d)) for i, d in enumerate(a + b)])
    assert any("estudios distintos" in w for w in warns)


# ----------------------------------------------------------------------------- preprocesamiento
def test_model_input_matches_training_contract():
    dsets = fake_study(60)
    rng = np.random.default_rng(0)
    vol, n_files, warns = core.model_input([dsets[i] for i in rng.permutation(60)])  # orden arbitrario
    assert vol.shape == (48, 3, 224, 224) and vol.dtype == np.uint8 and n_files == 60
    vol2, _, _ = core.model_input(dsets)
    assert np.array_equal(vol, vol2), "el orden de los archivos no debe cambiar la entrada"


def test_display_slice_uses_bone_window():
    hu, win = core.display_slice(_ct_slice())
    assert win.shape == (224, 224) and 0 <= win.min() and win.max() <= 1
    assert hu.shape == (128, 128)


def test_sort_by_z():
    d = fake_study(5)
    out = core.sort_by_z([d[3], d[0], d[4], d[1], d[2]])
    assert [int(x.InstanceNumber) for x in out] == [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------------- inferencia (mecánica)
@pytest.mark.parametrize("name", ["mil", "bigru"])
def test_inference_code_path_with_random_weights(tmp_path, name):
    """Prueba de mecánica: checkpoint con pesos ALEATORIOS en tmp; no es un resultado."""
    import torch
    import common
    import inference
    from s04_train_eval import build_model
    ck = tmp_path / f"{name}_aleatorio.pt"
    torch.save(dict(state_dict=build_model(name, pretrained=False).state_dict(), model_name=name,
                    threshold=0.5, preprocessing=common.preprocessing_config()), ck)
    clf = inference.load_classifier(ck, device="cpu")
    out = core.predict(fake_study(50), {name: clf})[name]
    assert 0 <= out["probability"] <= 1 and out["prediction"] in (0, 1)
    assert out["input_shape"] == [48, 3, 224, 224]
    assert (out["slice_probabilities"] is not None) == (name == "mil")


# ----------------------------------------------------------------------------- app completa
def test_app_starts_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app_streamlit.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8
    # sin checkpoints: la predicción debe aparecer como PENDIENTE, nunca simulada
    if any(core.checkpoint_status(m, verify_hash=False)["estado"] == "PENDIENTE" for m in core.MODELS):
        assert any("PENDIENTE" in w.value for w in at.warning)


def test_app_toggles_static_mode_and_hidden_charts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app_streamlit.py"), default_timeout=120).run()
    at.toggle[0].set_value(False).run()
    assert not at.exception
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception


# ----------------------------------------------------------------------------- comparación (lógica)
def test_evaluation_lookup_and_percentiles():
    pr = core.predictions("bigru")
    uid = pr.StudyInstanceUID.iloc[0]
    ev = core.evaluation_lookup(uid)
    assert ev["split"] == "test" and set(ev["prob"]) == {"mil", "bigru"}
    assert abs(ev["prob"]["bigru"] - pr.prob.iloc[0]) < 1e-9
    assert core.evaluation_lookup("1.2.3.4") is None
    pct = core.test_percentile("bigru", 1.01)
    assert pct["global_"] == 100.0


def test_metrics_ci_contains_point_estimate():
    pr, mt = core.predictions("bigru"), core.metrics("bigru")["test_thr_val_youden"]
    ci = core.metrics_ci(pr.patient_overall.values, pr.prob.values, mt["threshold"], n_resamples=300)
    for k in ("sensitivity", "specificity", "roc_auc"):
        assert ci[k][0] <= mt[k] <= ci[k][1]


def test_compare_results_flags_non_reproduction():
    uid = core.predictions("bigru").StudyInstanceUID.iloc[0]
    ev = core.evaluation_lookup(uid)
    fake = {m: dict(probability=ev["prob"][m], threshold=.5, prediction=int(ev["prob"][m] >= .5)) for m in core.MODELS}
    table, summary = core.compare_results(fake, uid)
    assert table.reproduce_evaluacion.all() and "acuerdo" in summary
    fake["bigru"]["probability"] += 0.05
    table, _ = core.compare_results(fake, uid)
    assert not table.set_index("model").loc["bigru", "reproduce_evaluacion"]


# ----------------------------------------------------------------------------- app con checkpoints
@pytest.fixture
def fake_checkpoints(tmp_path, monkeypatch):
    """Checkpoints con pesos ALEATORIOS + MANIFEST propio de la prueba + un estudio local cuyo UID
    pertenece a la partición de prueba. Solo verifica la interfaz; las salidas no son resultados."""
    import json
    import torch
    import common
    from s04_train_eval import build_model
    weights, dicom = tmp_path / "weights", tmp_path / "dicom"
    weights.mkdir()
    files = []
    for m in core.MODELS:
        torch.manual_seed(0)
        p = weights / f"{m}_final.pt"
        torch.save(dict(state_dict=build_model(m, pretrained=False).state_dict(), model_name=m, threshold=0.5,
                        preprocessing=common.preprocessing_config()), p)
        files.append(dict(path=f"weights/{m}_final.pt", sha256=core.sha256(p)))
    for rel in ("results/test_slice_probs_mil.npz",):
        files.append(core._manifest_entry(rel))
    (tmp_path / "MANIFEST.json").write_text(json.dumps(dict(files=files)))
    uid = core.predictions("bigru").StudyInstanceUID.iloc[0]
    (dicom / uid).mkdir(parents=True)
    for i, d in enumerate(fake_study(50, uid)):
        d.save_as(dicom / uid / f"{i + 1}.dcm", enforce_file_format=True)
    monkeypatch.setattr(core, "WEIGHTS_DIR", weights)
    monkeypatch.setattr(core, "MANIFEST", tmp_path / "MANIFEST.json")
    monkeypatch.setattr(common, "DICOM_DIR", dicom)
    return uid


def test_app_prediction_and_comparison_flow(fake_checkpoints):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app_streamlit.py"), default_timeout=300).run()
    assert not at.exception
    tab = at.tabs[3]
    tab.radio[1].set_value("Un estudio").run()          # modo
    at.tabs[3].radio[2].set_value("Estudio local (experimento_dl/data/dicom/)").run()
    btn = [b for b in at.tabs[3].button if b.label == "Ejecutar predicción"]
    assert btn, "el botón de predicción debe aparecer cuando hay checkpoints válidos"
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    txt = " ".join(e.value for e in list(at.success) + list(at.warning) + list(at.error) + list(at.info))
    assert "coinciden" in txt or "DISCREPAN" in txt                    # acuerdo A vs B
    assert "NO reproduce" in txt                                       # pesos aleatorios ≠ evaluación
    assert "patient_overall" in txt                                    # etiqueta real rotulada aparte
    assert len(at.tabs[3].dataframe) >= 1


def test_batch_inference(fake_checkpoints):
    import inference
    clfs = {m: inference.load_classifier(core.checkpoint_path(m), device="cpu") for m in core.MODELS}
    df = core.predict_local_batch([fake_checkpoints, "no-existe"], clfs)
    assert set(df.model.dropna()) == {"mil", "bigru"}
    assert df.error.notna().sum() == 1                       # el estudio inexistente se informa, no rompe el lote


def test_validation_script_detects_non_reproduction(fake_checkpoints, tmp_path):
    import json
    sys.path.insert(0, str(ROOT / "tools"))
    import validar_inferencia
    with pytest.raises(SystemExit) as e:
        validar_inferencia.main(["--out", str(tmp_path / "out")])
    assert e.value.code == 1                                   # pesos aleatorios: no reproduce la evaluación
    s = json.loads((tmp_path / "out" / "validacion_inferencia.json").read_text())
    assert s["reproducibilidad"]["n"] == 2 and s["reproducibilidad"]["reproduce"] == 0


def test_missing_modality_is_not_flagged():
    d = fake_study(2)
    for x in d:
        del x.Modality
    _, _, warns = core.read_dicoms([Upload(f"{i}.dcm", to_bytes(x)) for i, x in enumerate(d)])
    assert not any("Modalidad" in w for w in warns)
