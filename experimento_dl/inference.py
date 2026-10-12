"""Inferencia reutilizable (contrato de integración con la app Streamlit).

    from inference import load_classifier, predict_study
    clf = load_classifier("data/weights/mil_final.pt")          # o bigru_final.pt
    out = predict_study(["/ruta/1.dcm", "/ruta/2.dcm", ...], clf)  # rutas, bytes o file-like
    out["probability"], out["prediction"]

- Entrada: TODOS los cortes axiales de UN estudio de TC (sin etiquetas). Se aceptan rutas,
  objetos file-like (p. ej. st.file_uploader) o pydicom.Dataset.
- Preprocesamiento: exactamente el de entrenamiento, reconstruido a partir de la configuración
  guardada en el checkpoint (K, 2D/2.5D, ventana, tamaño) y de las mismas funciones de s03.
- Salida: dict con probability (0-1), prediction (0/1 según el umbral elegido en validación),
  threshold, model, n_files, n_positions, slice_probabilities (solo MIL), warnings.
- No usa etiquetas, no localiza fracturas y no tiene validación clínica.
Requiere que las variables de entorno RSNA_EXP/K_SLICES/CONTEXT_MM NO contradigan el checkpoint:
la configuración se fija aquí a partir del checkpoint antes de importar los módulos del pipeline.
"""
import io
import os
from pathlib import Path

import numpy as np
import torch


def load_classifier(checkpoint_path, device=None):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pre = ck["preprocessing"]
    # Fijar la configuración del pipeline ANTES de importar common/s03/s04
    os.environ["RSNA_EXP"] = pre["exp"]
    os.environ["K_SLICES"] = str(pre["k_slices"])
    os.environ["CONTEXT_MM"] = str(pre["context_mm"])
    import common
    if (common.K_SLICES, common.CONTEXT_MM) != (pre["k_slices"], pre["context_mm"]):
        raise RuntimeError("common ya estaba importado con otra configuración; usar un proceso nuevo "
                           f"(checkpoint K={pre['k_slices']}, context={pre['context_mm']}).")
    from s04_train_eval import build_model
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(ck["model_name"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return dict(model=model, device=device, name=ck["model_name"], threshold=float(ck["threshold"]),
                preprocessing=pre, checkpoint=str(checkpoint_path))


def _read(item):
    import pydicom
    if isinstance(item, pydicom.Dataset):
        return item
    if isinstance(item, (str, Path)):
        return pydicom.dcmread(str(item))
    if isinstance(item, (bytes, bytearray)):
        return pydicom.dcmread(io.BytesIO(item))
    data = item.getvalue() if hasattr(item, "getvalue") else item.read()
    return pydicom.dcmread(io.BytesIO(data))


def preprocess_study(items, preprocessing):
    """Devuelve (vol uint8 [K,H,W] o [K,3,H,W], n_archivos, avisos). Mismas funciones que s03."""
    from s03_preprocess import build_volume, sample_25d, select_uploaded, slice_record
    warnings = []
    dsets = []
    for it in items:
        try:
            dsets.append(_read(it))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"archivo no legible: {type(e).__name__}")
    if not dsets:
        raise ValueError("ningún DICOM legible")
    uids = {getattr(d, "StudyInstanceUID", None) for d in dsets}
    if len(uids) > 1:
        warnings.append(f"se recibieron cortes de {len(uids)} estudios distintos")
    if any(not hasattr(d, "RescaleSlope") for d in dsets):
        warnings.append("algún corte sin RescaleSlope/Intercept: escala HU no garantizada")
    dsets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0))
    k = preprocessing["k_slices"]
    if len(dsets) < k:
        warnings.append(f"solo {len(dsets)} cortes (< {k}); se rellenan repitiendo el último")
    if preprocessing["context_mm"] > 0:
        vol, _, _ = sample_25d(len(dsets), lambda i, h: dsets[i], "inferencia",
                               lambda i: f"inst_{getattr(dsets[i], 'InstanceNumber', i)}")
    else:
        sel = select_uploaded(dsets)
        recs = [slice_record(d, f"inst_{getattr(d, 'InstanceNumber', i)}", "inferencia", i + 1)
                for i, d in enumerate(sel)]
        vol, _ = build_volume(recs)
    return vol, len(dsets), warnings


@torch.no_grad()
def predict_volume(vol, clf):
    from s04_train_eval import to_input
    dev = clf["device"]
    x = torch.from_numpy(vol[None])
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        logit, slog = clf["model"](to_input(x, dev, train=False))
    p = float(torch.sigmoid(logit.float())[0])
    sp = None if slog is None else torch.sigmoid(slog.float())[0].cpu().numpy().round(4).tolist()
    return p, sp


def predict_study(items, clf):
    vol, n_files, warnings = preprocess_study(items, clf["preprocessing"])
    p, sp = predict_volume(vol, clf)
    return dict(probability=p, prediction=int(p >= clf["threshold"]), threshold=clf["threshold"],
                model=clf["name"], n_files=n_files, n_positions=int(vol.shape[0]),
                input_shape=list(vol.shape), slice_probabilities=sp, warnings=warnings,
                disclaimer="Clasificación global por estudio; no localiza fracturas; sin validación clínica.")


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Inferencia sobre un directorio con los .dcm de un estudio")
    ap.add_argument("checkpoint")
    ap.add_argument("study_dir")
    a = ap.parse_args()
    c = load_classifier(a.checkpoint)
    files = sorted(Path(a.study_dir).glob("*.dcm"))
    r = predict_study(files, c)
    r.pop("slice_probabilities")
    print(json.dumps(r, indent=2, ensure_ascii=False))
