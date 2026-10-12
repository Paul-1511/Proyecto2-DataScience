"""Fase 3: DICOM -> HU -> ventana ósea -> [0,1] -> 224x224, cortes ordenados por z.

v1 (2D):   data/cache/<UID>.npy      uint8 [K,224,224]   (K=24)
v2 (2.5D): <CACHE_DIR>/<UID>.npz     uint8 [K,3,224,224] (K=48), clave "vol", comprimido.
  En 2.5D cada posición muestreada aporta 3 canales: el corte central y sus vecinos a
  ±CONTEXT_MM (desplazamiento en número de cortes = round(CONTEXT_MM / espaciado z)),
  ordenados por z. Las posiciones se eligen sobre la lista de cortes ordenada por número de
  instancia (misma regla 5%-95% equiespaciada) y el volumen final se ordena por z.
Se audita en results*/dicom_metadata.csv qué parámetros tenía cada DICOM (no se asumen iguales)
y se registran errores por estudio en preprocess_failures.csv.
Uso: python s03_preprocess.py [--workers N]     (RSNA_EXP=v2 para la configuración 2.5D)
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

from common import (BONE_WL, BONE_WW, CACHE_DIR, CONTEXT_MM, DICOM_DIR, FIG_DIR,  # noqa: E402
                    IMG_SIZE, K_SLICES, RESULTS_DIR, select_slice_numbers)

IS_25D = CONTEXT_MM > 0

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


def _window_resize(hu):
    return cv2.resize(bone_window(hu), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def plan_25d(n, spacing_mm, k=None):
    """Posiciones (0-based, sobre cortes ordenados por instancia) de los centros y desplazamiento
    de vecinos. Regla independiente de las etiquetas: aplicable a estudios nuevos."""
    centers = [p - 1 for p in select_slice_numbers(n, k)]
    step = max(1, int(round(CONTEXT_MM / spacing_mm))) if spacing_mm and spacing_mm > 0 else 1
    return centers, step


def sample_25d(n, get_ds, uid, name_of):
    """Núcleo 2.5D común a entrenamiento e inferencia.
    n: nº de cortes ordenados por InstanceNumber; get_ds(i, header_only) -> pydicom Dataset.
    Devuelve (vol uint8 [K,3,H,W] ordenado por z, registros de los centros ordenados por z, step)."""
    mid = max(0, min(n - 2, n // 2))
    z_a = z_position(get_ds(mid, True), mid + 1)
    z_b = z_position(get_ds(mid + 1, True), mid + 2) if n > 1 else z_a
    spacing = abs(z_b - z_a)
    centers, step = plan_25d(n, spacing)
    cache = {}

    def rec(i):
        i = min(max(i, 0), n - 1)
        if i not in cache:
            cache[i] = slice_record(get_ds(i, False), name_of(i), uid, i + 1)
        return cache[i]

    triplets = []
    for c in centers:
        trio = sorted([rec(c - step), rec(c), rec(c + step)], key=lambda r: r["z"])
        triplets.append((rec(c), trio))
    triplets.sort(key=lambda t: t[0]["z"])                     # orden anatómico por z
    vol = np.stack([np.stack([_window_resize(r["hu"]) for r in trio]) for _, trio in triplets])
    if len(vol) < K_SLICES:                                    # estudios con menos de K cortes
        vol = np.concatenate([vol, np.repeat(vol[-1:], K_SLICES - len(vol), 0)])
    vol = (vol[:K_SLICES] * 255).round().astype(np.uint8)
    recs = [c for c, _ in triplets]
    for r in recs:
        r["meta"].update(spacing_z_mm=round(spacing, 4), context_step=step)
    return vol, recs, step


def cache_path(uid):
    return CACHE_DIR / (f"{uid}.npz" if IS_25D else f"{uid}.npy")


def load_cached(uid):
    p = cache_path(uid)
    return np.load(p)["vol"] if p.suffix == ".npz" else np.load(p)


def process_study_25d(uid):
    files = sorted((DICOM_DIR / uid).glob("*.dcm"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"sin archivos .dcm en {DICOM_DIR / uid}")
    vol, recs, _ = sample_25d(len(files), lambda i, h: pydicom.dcmread(files[i], stop_before_pixels=h),
                              uid, lambda i: files[i].name)
    np.savez_compressed(cache_path(uid), vol=vol)
    for r in recs:
        r["meta"]["n_files_study"] = len(files)
    return [r["meta"] for r in recs], recs


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
    vol = load_cached(uid)
    if vol.ndim == 4:  # 2.5D: se muestra el canal central
        vol = vol[:, 1]
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


def save_25d_example(uid, label):
    vol = load_cached(uid)
    i = len(vol) // 2
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for c, t in enumerate([f"vecino -{CONTEXT_MM:g} mm", "corte central", f"vecino +{CONTEXT_MM:g} mm"]):
        ax[c].imshow(vol[i, c], cmap="gray")
        ax[c].set_title(t)
    ax[3].imshow(np.moveaxis(vol[i], 0, -1))
    ax[3].set_title("entrada RGB 2.5D")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Entrada 2.5D, posición {i}/{len(vol)} - patient_overall={label}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"entrada_25d_y{label}_{uid.split('.')[-1]}.png", dpi=100)
    plt.close(fig)


def _worker(uid):
    """Proceso independiente por estudio; los errores se devuelven, no detienen el lote."""
    import time
    t0 = time.time()
    try:
        meta, _ = (process_study_25d if IS_25D else process_study)(uid)
        return uid, meta, None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return uid, None, f"{type(e).__name__}: {e}"[:300], time.time() - t0


def main():
    """Incremental: procesa solo estudios disponibles que aún no estén en caché."""
    import argparse
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--limit", type=int, default=0, help="solo ~N estudios repartidos por partición (prueba)")
    args = ap.parse_args()
    splits = pd.read_csv(RESULTS_DIR / "splits.csv")
    if args.limit:
        splits = splits.groupby("split", group_keys=False).head(max(1, args.limit // 3))
    if "RSNA_DICOM_DIR" in os.environ:  # dataset completo montado: todos los estudios disponibles
        ok = {p.name for p in DICOM_DIR.iterdir() if p.is_dir()}
    else:
        log = pd.read_csv(RESULTS_DIR / "download_log.csv")
        ok = set(log.loc[log["status"] == "ok", "StudyInstanceUID"])
    n_examples = {lab: len(list(FIG_DIR.glob(f"preproc_antes_despues_y{lab}_*.png"))) for lab in (0, 1)}
    labels = splits.set_index("StudyInstanceUID")["patient_overall"].to_dict()
    todo = [u for u in splits["StudyInstanceUID"]
            if u in ok and not ((CACHE_DIR / f"{u}_meta.csv").exists() and cache_path(u).exists())]
    print(f"{len(todo)} estudios por procesar con {args.workers} procesos "
          f"({'2.5D' if IS_25D else '2D'}, K={K_SLICES})", flush=True)
    new, failed, times, t0 = 0, [], [], time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_worker, u) for u in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            uid, meta, err, dt = fut.result()
            times.append(dict(StudyInstanceUID=uid, seconds=round(dt, 2), ok=err is None))
            if err:
                failed.append(dict(StudyInstanceUID=uid, error=err))
            else:
                pd.DataFrame(meta).to_csv(CACHE_DIR / f"{uid}_meta.csv", index=False)
                new += 1
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  [{i}/{len(todo)}] {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.1f} min, "
                      f"fallidos {len(failed)}", flush=True)
    # figuras representativas (2 por clase), tras el lote para no depender del orden de llegada
    for uid in [u for u in splits["StudyInstanceUID"] if cache_path(u).exists()]:
        lab = int(labels[uid])
        if n_examples[lab] >= 2:
            continue
        if IS_25D:
            _, recs = process_study_25d(uid)
            save_25d_example(uid, lab)
        else:
            _, recs = process_study(uid)
        save_before_after(uid, recs, lab)
        save_montage(uid, lab)
        n_examples[lab] += 1
    tp = RESULTS_DIR / "preprocess_times.csv"
    if times:
        pd.DataFrame(times).to_csv(tp, mode="a", header=not tp.exists(), index=False)

    metas = [pd.read_csv(p) for p in sorted(CACHE_DIR.glob("*_meta.csv"))]
    meta = pd.concat(metas, ignore_index=True)
    meta.to_csv(RESULTS_DIR / "dicom_metadata.csv", index=False)
    audit = pd.DataFrame([z_audit(u, g.to_dict("records")) for u, g in meta.groupby("StudyInstanceUID")])
    audit.to_csv(RESULTS_DIR / "orden_z_auditoria.csv", index=False)
    fail_path = RESULTS_DIR / "preprocess_failures.csv"
    prev = pd.read_csv(fail_path) if fail_path.exists() else pd.DataFrame(columns=["StudyInstanceUID", "error"])
    allf = pd.concat([prev, pd.DataFrame(failed, columns=["StudyInstanceUID", "error"])])
    allf = allf[~allf["StudyInstanceUID"].map(lambda u: cache_path(u).exists())]  # ya recuperados
    allf = allf.reindex(columns=["StudyInstanceUID", "error"]).drop_duplicates("StudyInstanceUID", keep="last")
    allf.to_csv(fail_path, index=False)
    print(f"Nuevos: {new}  total en caché: {meta['StudyInstanceUID'].nunique()}  fallidos: {len(failed)}")
    print("Auditoría z:", audit["z_unique"].sum(), "con z únicos;",
          audit["z_monotonic_vs_instance"].sum(), "monótonos;",
          audit["z_direction_vs_instance"].value_counts().to_dict())
    for c in ["slope", "intercept", "is_hu", "rows", "photometric", "transfer_syntax"]:
        print(f"  {c}: {meta[c].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
