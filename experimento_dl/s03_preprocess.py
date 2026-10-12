"""Fase 3: DICOM -> HU -> ventana ósea -> [0,1] -> 224x224, cortes ordenados por z.

Por estudio se guarda data/cache/<UID>.npy (uint8, [K,224,224]) y se audita en
results/dicom_metadata.csv qué parámetros tenía cada DICOM (no se asumen iguales).
"""
import os
import warnings

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pydicom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (BONE_WL, BONE_WW, CACHE_DIR, DICOM_DIR, FIG_DIR, IMG_SIZE,  # noqa: E402
                    K_SLICES, RESULTS_DIR)

warnings.filterwarnings("ignore", category=UserWarning)


def to_hu(ds):
    """Aplica RescaleSlope/Intercept si existen; registra si se pudo obtener HU."""
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    has_rescale = hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept")
    # Valores de padding (fuera del FOV) -> aire
    pad = getattr(ds, "PixelPaddingValue", None)
    if pad is not None:
        img[img == float(pad)] = np.nan
    hu = img * slope + intercept
    hu = np.nan_to_num(hu, nan=-1024.0)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        hu = hu.max() + hu.min() - hu
    is_hu = has_rescale and getattr(ds, "Modality", "CT") == "CT"
    return hu, slope, intercept, is_hu


def bone_window(hu, wl=BONE_WL, ww=BONE_WW):
    lo, hi = wl - ww / 2, wl + ww / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)  # normalizado a [0,1]


def z_position(ds, fallback):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", fallback))


def slice_record(ds, name, uid, fallback_n):
    """Un corte DICOM -> HU + metadatos auditados (se leen por archivo, no se asumen)."""
    hu, slope, intercept, is_hu = to_hu(ds)
    z = z_position(ds, fallback_n)
    return dict(file=name, z=z, hu=hu,
                meta=dict(StudyInstanceUID=uid, file=name, slope=slope, intercept=intercept,
                          is_hu=is_hu, rows=int(ds.Rows), cols=int(ds.Columns),
                          bits=int(getattr(ds, "BitsStored", -1)),
                          pixel_representation=int(getattr(ds, "PixelRepresentation", -1)),
                          padding_value=getattr(ds, "PixelPaddingValue", None),
                          photometric=str(getattr(ds, "PhotometricInterpretation", "")),
                          transfer_syntax=str(ds.file_meta.TransferSyntaxUID.name),
                          pixel_spacing=str([float(v) for v in getattr(ds, "PixelSpacing", [])]),
                          slice_thickness=getattr(ds, "SliceThickness", None),
                          instance_number=getattr(ds, "InstanceNumber", None), z=z))


def build_volume(recs):
    """Ordena por z (ImagePositionPatient), ventana ósea, [0,1], 224x224, K fijo -> uint8.
    Misma función para entrenamiento y para la app (sin divergencias de preprocesamiento)."""
    recs = sorted(recs, key=lambda r: r["z"])
    vol = np.stack([cv2.resize(bone_window(r["hu"]), (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_AREA) for r in recs])
    # Rellenar si faltó algún corte (repetición del último) para tener K fijo
    if len(vol) < K_SLICES:
        vol = np.concatenate([vol, np.repeat(vol[-1:], K_SLICES - len(vol), 0)])
    return (vol[:K_SLICES] * 255).round().astype(np.uint8), recs


def select_uploaded(datasets):
    """Para un estudio completo subido en la app: ordena por InstanceNumber y aplica la misma
    regla de muestreo (select_slice_numbers sobre posiciones 1..N). Si se suben <= K cortes
    (p. ej. los 24 ya muestreados), se usan todos."""
    from common import select_slice_numbers
    datasets = sorted(datasets, key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0))
    if len(datasets) <= K_SLICES:
        return datasets
    return [datasets[i - 1] for i in select_slice_numbers(len(datasets))]


def process_study(uid):
    files = sorted((DICOM_DIR / uid).glob("*.dcm"), key=lambda p: int(p.stem))
    if len(files) > K_SLICES:  # estudio completo (p. ej. Kaggle Notebook): misma regla de muestreo
        from common import select_slice_numbers
        sel = set(select_slice_numbers(int(files[-1].stem)))
        files = [f for f in files if int(f.stem) in sel]
    recs = [slice_record(pydicom.dcmread(f), f.name, uid, int(f.stem)) for f in files]
    vol, recs = build_volume(recs)
    np.save(CACHE_DIR / f"{uid}.npy", vol)
    return [r["meta"] for r in recs], recs


def save_before_after(uid, recs, label):
    mid = recs[len(recs) // 2]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    a = ax[0].imshow(mid["hu"], cmap="gray")
    ax[0].set_title(f"HU crudo ({mid['hu'].shape[0]}x{mid['hu'].shape[1]})")
    plt.colorbar(a, ax=ax[0], fraction=0.046)
    ax[1].hist(mid["hu"].ravel(), bins=100, color="gray")
    ax[1].axvspan(BONE_WL - BONE_WW / 2, BONE_WL + BONE_WW / 2, color="orange", alpha=.25,
                  label=f"ventana ósea L{BONE_WL}/W{BONE_WW}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("HU"); ax[1].legend(); ax[1].set_title("Histograma HU")
    ax[2].imshow(cv2.resize(bone_window(mid["hu"]), (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_AREA), cmap="gray", vmin=0, vmax=1)
    ax[2].set_title(f"Ventana ósea + {IMG_SIZE}x{IMG_SIZE}")
    for x in (ax[0], ax[2]):
        x.axis("off")
    fig.suptitle(f"Estudio ...{uid[-12:]}  patient_overall={label}  corte z={mid['z']:.1f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"preproc_antes_despues_y{label}_{uid.split('.')[-1]}.png", dpi=110)
    plt.close(fig)


def save_montage(uid, label):
    vol = np.load(CACHE_DIR / f"{uid}.npy")
    cols = 8
    rows = int(np.ceil(len(vol) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i, a in enumerate(ax.ravel()):
        a.axis("off")
        if i < len(vol):
            a.imshow(vol[i], cmap="gray")
            a.set_title(str(i), fontsize=7)
    fig.suptitle(f"{len(vol)} cortes muestreados (orden z) - patient_overall={label}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"montaje_cortes_y{label}_{uid.split('.')[-1]}.png", dpi=90)
    plt.close(fig)


def z_audit(uid, meta):
    """Consistencia del orden por ImagePositionPatient: z únicos, monótono respecto al
    número de instancia y espaciado entre cortes muestreados de signo constante."""
    m = pd.DataFrame(meta).assign(n=lambda d: d["file"].str[:-4].astype(int)).sort_values("n")
    dz = np.diff(m["z"].values)
    return dict(StudyInstanceUID=uid, n_files=len(m), z_unique=bool(m["z"].is_unique),
                z_monotonic_vs_instance=bool((dz > 0).all() or (dz < 0).all()),
                z_direction_vs_instance="decreciente" if (dz < 0).all() else
                ("creciente" if (dz > 0).all() else "mixto"),
                z_min=float(m["z"].min()), z_max=float(m["z"].max()),
                dz_cv=float(np.std(np.abs(dz)) / np.mean(np.abs(dz))) if len(dz) else np.nan)


def main():
    """Incremental: procesa solo estudios descargados ('ok') que aún no estén en caché."""
    splits = pd.read_csv(RESULTS_DIR / "splits.csv")
    if "RSNA_DICOM_DIR" in os.environ:  # dataset completo montado: todos los estudios disponibles
        ok = {p.name for p in DICOM_DIR.iterdir() if p.is_dir()}
    else:
        log = pd.read_csv(RESULTS_DIR / "download_log.csv")
        ok = set(log.loc[log["status"] == "ok", "StudyInstanceUID"])
    n_examples = {lab: len(list(FIG_DIR.glob(f"preproc_antes_despues_y{lab}_*.png"))) for lab in (0, 1)}
    new, failed = 0, []
    for _, row in splits.iterrows():
        uid, lab = row["StudyInstanceUID"], int(row["patient_overall"])
        meta_path = CACHE_DIR / f"{uid}_meta.csv"
        if uid not in ok or (meta_path.exists() and (CACHE_DIR / f"{uid}.npy").exists()):
            continue
        try:
            meta, recs = process_study(uid)
        except Exception as e:  # noqa: BLE001
            failed.append(dict(StudyInstanceUID=uid, error=repr(e)[:200]))
            continue
        pd.DataFrame(meta).to_csv(meta_path, index=False)
        new += 1
        if n_examples[lab] < 2:
            save_before_after(uid, recs, lab)
            save_montage(uid, lab)
            n_examples[lab] += 1

    metas = [pd.read_csv(p) for p in sorted(CACHE_DIR.glob("*_meta.csv"))]
    meta = pd.concat(metas, ignore_index=True)
    meta.to_csv(RESULTS_DIR / "dicom_metadata.csv", index=False)
    audit = pd.DataFrame([z_audit(u, g.to_dict("records")) for u, g in meta.groupby("StudyInstanceUID")])
    audit.to_csv(RESULTS_DIR / "orden_z_auditoria.csv", index=False)
    pd.DataFrame(failed, columns=["StudyInstanceUID", "error"]).to_csv(
        RESULTS_DIR / "preprocess_failures.csv", index=False)
    print(f"Nuevos: {new}  total en caché: {meta['StudyInstanceUID'].nunique()}  fallidos: {len(failed)}")
    print("Auditoría z:", audit["z_unique"].sum(), "con z únicos;",
          audit["z_monotonic_vs_instance"].sum(), "monótonos;",
          audit["z_direction_vs_instance"].value_counts().to_dict())
    for c in ["slope", "intercept", "is_hu", "rows", "photometric", "transfer_syntax"]:
        print(f"  {c}: {meta[c].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
