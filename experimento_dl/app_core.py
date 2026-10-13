"""Lógica reutilizable de la app Streamlit (sin dependencias de Streamlit).

La usan `app_streamlit.py`, `notebooks/04_integracion_app.ipynb` y `tests/test_app_core.py`.

- Fija la configuración v2 (RSNA_EXP=v2, artefactos en artifacts/v2/) ANTES de importar `common`,
  de modo que `inference.load_classifier` no encuentre un conflicto de configuración.
- Lee solo artefactos reales producidos por el entrenamiento (artifacts/v2/). Si un archivo no
  existe, las funciones devuelven None / estado PENDIENTE; nunca generan valores sustitutos.
- No reimplementa el preprocesamiento ni las arquitecturas: usa `inference.py` y `s03_preprocess`.
"""
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
V2_DIR = Path(os.environ.get("RSNA_ARTIFACTS_DIR", REPO / "artifacts" / "v2"))

os.environ.setdefault("RSNA_EXP", "v2")
os.environ.setdefault("RSNA_RESULTS_DIR", str(V2_DIR / "results"))
os.environ.setdefault("RSNA_WEIGHTS_DIR", str(V2_DIR / "weights"))
if os.environ["RSNA_EXP"] != "v2":
    raise RuntimeError("app_core requiere RSNA_EXP=v2 (los modelos entrenados son v2).")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

RESULTS_DIR = common.RESULTS_DIR
WEIGHTS_DIR = common.WEIGHTS_DIR
FIG_DIR = common.FIG_DIR
CONFIG_DIR = V2_DIR / "config"
MANIFEST = V2_DIR / "MANIFEST.json"

MODELS = ("mil", "bigru")
NAMES = {"mil": "Modelo A · ResNet18 + MIL (top-3)", "bigru": "Modelo B · ResNet18 + BiGRU"}
SHORT = {"mil": "A · MIL", "bigru": "B · BiGRU"}
SETS = {"val": "Validación (n=323)", "test_thr_val_youden": "Prueba · umbral de validación",
        "test_thr_0_5": "Prueba · umbral 0.5"}
METRIC_KEYS = ["roc_auc", "sensitivity", "specificity", "precision", "f1", "accuracy"]
METRIC_LABELS = {"roc_auc": "AUC ROC", "sensitivity": "Sensibilidad", "specificity": "Especificidad",
                 "precision": "Precisión (VPP)", "f1": "F1", "accuracy": "Exactitud"}

MAX_UPLOAD_FILES = 2000
DISCLAIMER = "Clasificación global por estudio; no localiza fracturas; sin validación clínica."


# ----------------------------------------------------------------------------- artefactos
def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_csv(path, **kw):
    path = Path(path)
    return pd.read_csv(path, **kw) if path.exists() else None


def preprocessing_config():
    """Configuración de preprocesamiento v2 publicada por entrenamiento (config/preprocessing.json)."""
    return read_json(CONFIG_DIR / "preprocessing.json")


def model_config(m):
    return read_json(CONFIG_DIR / f"model_{m}.json")


def metrics(m):
    return read_json(RESULTS_DIR / f"metrics_{m}.json")


def predictions(m, split="test"):
    return read_csv(RESULTS_DIR / f"{split}_predictions_{m}.csv")


def history(m):
    return read_csv(RESULTS_DIR / f"history_{m}.csv")


def comparison_table():
    return read_csv(RESULTS_DIR / "comparacion_modelos.csv")


def paired_comparison():
    return read_json(RESULTS_DIR / "comparacion_pareada.json")


def splits():
    return read_csv(RESULTS_DIR / "splits.csv")


def dicom_metadata():
    return read_csv(RESULTS_DIR / "dicom_metadata.csv.gz")


def mil_test_slice_probs():
    """(uids, probs [n, 48]) de test_slice_probs_mil.npz, o None si no existe."""
    p = RESULTS_DIR / "test_slice_probs_mil.npz"
    if not p.exists():
        return None
    # Los UID se guardaron como arreglo de objetos (requiere pickle): solo se carga el archivo
    # publicado, verificado contra el SHA-256 de MANIFEST.json.
    ref = _manifest_entry("results/test_slice_probs_mil.npz")
    if ref is None or sha256(p) != ref["sha256"]:
        return None
    z = np.load(p, allow_pickle=True)
    return [str(u) for u in z["uids"]], z["probs"]


def figures(prefix=""):
    return sorted(FIG_DIR.glob(f"{prefix}*.png")) if FIG_DIR.exists() else []


def metrics_long():
    """Tabla larga modelo × conjunto × métrica, a partir de metrics_{m}.json (para gráficas)."""
    rows = []
    for m in MODELS:
        mt = metrics(m)
        if mt is None:
            continue
        for s, label in SETS.items():
            for k in METRIC_KEYS:
                rows.append(dict(modelo=SHORT[m], model=m, conjunto=label, set=s, metrica=METRIC_LABELS[k],
                                 key=k, valor=float(mt[s][k]), umbral=float(mt[s]["threshold"])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- checkpoints
def _manifest_entry(rel):
    man = read_json(MANIFEST) or {}
    return next((f for f in man.get("files", []) if f["path"] == rel), None)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_path(m):
    return WEIGHTS_DIR / f"{m}_final.pt"


def checkpoint_status(m, verify_hash=True):
    """Estado del checkpoint de un modelo.
    estado: DISPONIBLE (existe y coincide con MANIFEST), PENDIENTE (no existe) o
    HASH DISTINTO (existe pero no es el archivo publicado: no se usa)."""
    p = checkpoint_path(m)
    ref = _manifest_entry(f"weights/{m}_final.pt")
    out = dict(model=m, nombre=NAMES[m], ruta=str(p), existe=p.exists(),
               sha256_esperado=ref["sha256"] if ref else None, sha256=None)
    if not p.exists():
        out["estado"] = "PENDIENTE"
        return out
    out["bytes"] = p.stat().st_size
    if verify_hash and ref:
        out["sha256"] = sha256(p)
        out["estado"] = "DISPONIBLE" if out["sha256"] == ref["sha256"] else "HASH DISTINTO"
    else:
        out["estado"] = "DISPONIBLE"
    return out


def artifact_status():
    """Disponibilidad de cada artefacto que consume la app (para el panel de estado)."""
    rows = []
    for rel, who in [("config/preprocessing.json", "Entrenamiento (Majo)"),
                     ("results/metrics_mil.json", "Entrenamiento (Majo)"),
                     ("results/metrics_bigru.json", "Entrenamiento (Majo)"),
                     ("results/comparacion_modelos.csv", "Entrenamiento (Majo)"),
                     ("results/comparacion_pareada.json", "Entrenamiento (Majo)"),
                     ("results/test_predictions_mil.csv", "Entrenamiento (Majo)"),
                     ("results/test_predictions_bigru.csv", "Entrenamiento (Majo)"),
                     ("results/history_mil.csv", "Entrenamiento (Majo)"),
                     ("results/history_bigru.csv", "Entrenamiento (Majo)"),
                     ("weights/mil_final.pt", "Entrenamiento (Majo) · fuera de git"),
                     ("weights/bigru_final.pt", "Entrenamiento (Majo) · fuera de git")]:
        rows.append(dict(archivo=f"artifacts/v2/{rel}", produce=who,
                         estado="disponible" if (V2_DIR / rel).exists() else "PENDIENTE"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- métricas
def metrics_at_threshold(y, p, thr):
    """Matriz de confusión y métricas a un umbral dado, sobre predicciones reales guardadas."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    f1 = div(2 * prec * sens, prec + sens) if not np.isnan(prec) else float("nan")
    return dict(threshold=float(thr), n=len(y), tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=sens,
                specificity=spec, precision=prec, f1=f1, accuracy=div(tp + tn, len(y)))


def roc_points(y, p):
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thr)), float(roc_auc_score(y, p))


# ----------------------------------------------------------------------------- entradas DICOM
class InvalidInput(ValueError):
    """Entrada no utilizable (sin archivos, ninguno legible, sin píxeles…)."""


def _name(item, i):
    return getattr(item, "name", None) or (Path(item).name if isinstance(item, (str, Path)) else f"archivo_{i}")


def read_dicoms(items, max_files=MAX_UPLOAD_FILES):
    """Lee y valida los archivos de UN estudio. Devuelve (datasets, errores, avisos).
    Lanza InvalidInput si no queda ningún corte utilizable."""
    import pydicom
    if not items:
        raise InvalidInput("No se recibieron archivos.")
    if len(items) > max_files:
        raise InvalidInput(f"Demasiados archivos ({len(items)} > {max_files}); sube un único estudio.")
    dsets, errors, warnings = [], [], []
    for i, it in enumerate(items):
        name = _name(it, i)
        try:
            if isinstance(it, (str, Path)):
                ds = pydicom.dcmread(str(it))
            else:
                data = it if isinstance(it, (bytes, bytearray)) else (
                    it.getvalue() if hasattr(it, "getvalue") else it.read())
                if not data:
                    raise ValueError("archivo vacío")
                ds = pydicom.dcmread(io.BytesIO(data))
            if "PixelData" not in ds:
                raise ValueError("sin datos de píxel")
            dsets.append(ds)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")
    if not dsets:
        raise InvalidInput("Ningún archivo es un DICOM con imagen legible. " + "; ".join(errors[:3]))
    modalities = {str(getattr(d, "Modality", "?")) for d in dsets}
    if modalities - {"CT"}:
        warnings.append(f"Modalidad distinta de CT: {sorted(modalities)}. Los modelos se entrenaron solo con TC.")
    uids = {str(getattr(d, "StudyInstanceUID", "?")) for d in dsets}
    if len(uids) > 1:
        warnings.append(f"Se mezclan {len(uids)} estudios distintos; la predicción no sería válida.")
    k = (preprocessing_config() or {}).get("k_slices", common.K_SLICES)
    if len(dsets) < k:
        warnings.append(f"Solo {len(dsets)} cortes (< {k} posiciones del modelo): se repetirá el último "
                        "corte. Sube el estudio completo para una entrada equivalente al entrenamiento.")
    return dsets, errors, warnings


def sort_by_z(dsets):
    from s03_preprocess import z_position
    return sorted(dsets, key=lambda d: z_position(d, int(getattr(d, "InstanceNumber", 0) or 0)))


def study_summary(dsets):
    d0 = dsets[0]
    g = lambda k: str(getattr(d0, k, "—"))  # noqa: E731
    return {"StudyInstanceUID": g("StudyInstanceUID"), "Modalidad": g("Modality"), "Cortes": len(dsets),
            "Filas×Columnas": f"{g('Rows')}×{g('Columns')}", "Grosor de corte (mm)": g("SliceThickness"),
            "PixelSpacing": g("PixelSpacing"), "Fotometría": g("PhotometricInterpretation"),
            "RescaleSlope / Intercept": f"{g('RescaleSlope')} / {g('RescaleIntercept')}",
            "Transfer syntax": str(d0.file_meta.TransferSyntaxUID.name) if hasattr(d0, "file_meta") else "—"}


def display_slice(ds):
    """(HU crudo, corte con ventana ósea [0,1] a 224×224) con las funciones de s03."""
    from s03_preprocess import _window_resize, to_hu
    hu = to_hu(ds)[0]
    return hu, _window_resize(hu)


def model_input(dsets):
    """Tensor de entrada del modelo (uint8 [K,3,224,224]) con el preprocesamiento de inference.py.
    Solo necesita config/preprocessing.json, no los checkpoints."""
    import inference
    pre = preprocessing_config()
    if pre is None:
        raise FileNotFoundError("Falta artifacts/v2/config/preprocessing.json")
    return inference.preprocess_study(dsets, pre)


def local_studies(min_files=1):
    """Estudios DICOM locales (carpetas con .dcm en common.DICOM_DIR)."""
    d = common.DICOM_DIR
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and len(list(p.glob("*.dcm"))) >= min_files)


def local_study_files(uid):
    return sorted((common.DICOM_DIR / uid).glob("*.dcm"))


# ----------------------------------------------------------------------------- inferencia
def load_models(which):
    """Carga los checkpoints DISPONIBLES de `which`; devuelve {m: clf} y {m: motivo} de los omitidos."""
    import inference
    loaded, skipped = {}, {}
    for m in which:
        st = checkpoint_status(m)
        if st["estado"] != "DISPONIBLE":
            skipped[m] = st["estado"]
            continue
        try:
            loaded[m] = inference.load_classifier(checkpoint_path(m))
        except Exception as e:  # noqa: BLE001
            skipped[m] = f"error al cargar: {type(e).__name__}: {e}"
    return loaded, skipped


def predict(dsets, classifiers):
    """Predicción real con inference.predict_study para cada clasificador cargado."""
    import time
    import inference
    out = {}
    for m, clf in classifiers.items():
        t = time.time()
        r = inference.predict_study(dsets, clf)
        r["seconds"] = time.time() - t
        out[m] = r
    return out
