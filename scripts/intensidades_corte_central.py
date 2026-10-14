"""Sección 5 del notebook 02 (intensidades HU y antes/después) sin montar la competencia completa.

Reproduce la misma selección del notebook 02 sobre la auditoría v2 (2019 estudios): todos los estudios
con RescaleIntercept ∉ {−1024, 0} (hasta 6) más estudios aleatorios (semilla 42) hasta 24. Descarga por
la API de Kaggle solo el corte central de cada uno (archivo files[len//2] con nombres 1..N) y calcula
las mismas estadísticas y figuras con to_hu / _window_resize de s03_preprocess (solo importadas).

    .venv/bin/python scripts/intensidades_corte_central.py
Salidas: artifacts/eda_imagenes/auditoria_v2/intensidades_corte_central.csv y
figures/05_intensidades_hu.png, figures/06_antes_despues_preprocesamiento.png.
Requiere credenciales de Kaggle del usuario y reglas aceptadas. Los DICOM se guardan fuera de git.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "eda_imagenes" / "auditoria_v2"
FIG = OUT / "figures"
CACHE = REPO / "experimento_dl" / "data" / "corte_central"          # no versionado
COMP = "rsna-2022-cervical-spine-fracture-detection"
os.environ.update(RSNA_EXP="v2", RSNA_DICOM_DIR=str(CACHE), RSNA_CACHE_DIR=str(CACHE / "_c"),
                  RSNA_WEIGHTS_DIR=str(CACHE / "_w"), RSNA_RESULTS_DIR=str(CACHE / "_r"))
sys.path.insert(0, str(REPO / "experimento_dl"))


def download_central(api, uid, n):
    dest = CACHE / uid
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{n // 2 + 1}.dcm"                    # = sorted(files, key=int)[len // 2] con nombres 1..N
    if not (dest / name).exists():
        for intento in range(6):
            try:
                api.competition_download_file(COMP, f"train_images/{uid}/{name}", path=str(dest), quiet=True)
                break
            except Exception as e:  # noqa: BLE001
                time.sleep(10 * 2 ** intento if "429" in str(e) else 3)
        time.sleep(0.3)
    return dest / name


def main():
    import pydicom
    from kaggle.api.kaggle_api_extended import KaggleApi
    from s03_preprocess import _window_resize, to_hu
    api = KaggleApi()
    api.authenticate()
    studies = pd.read_csv(OUT / "metadata_studies.csv")
    labels = pd.read_csv(REPO / "train.csv")[["StudyInstanceUID", "patient_overall"]]
    rng = np.random.default_rng(42)
    atip = studies[~studies.intercept.isin([-1024.0, 0.0])].StudyInstanceUID.tolist()[:6]
    resto = [u for u in studies.StudyInstanceUID if u not in atip]
    pick = atip + list(rng.choice(resto, size=min(24 - len(atip), len(resto)), replace=False))
    n_of = studies.set_index("StudyInstanceUID").n_slices
    recs, fails, cache_hu = [], [], {}
    for u in pick:
        try:
            ds = pydicom.dcmread(str(download_central(api, u, int(n_of[u]))))
            hu, slope, icpt, _ = to_hu(ds)
            cache_hu[u] = hu
            grp = "intercept −8192/−10240 o slope≠1" if (slope != 1 or icpt not in (-1024.0, 0.0)) else f"intercept {int(icpt)}"
            recs.append(dict(StudyInstanceUID=u, grupo=grp, slope=slope, intercept=icpt, hu_min=hu.min(),
                             hu_p1=np.percentile(hu, 1), hu_p50=np.percentile(hu, 50), hu_p99=np.percentile(hu, 99),
                             hu_max=hu.max(), frac_aire=(hu < -900).mean(), frac_hueso=(hu > 300).mean(),
                             raw_max=int(ds.pixel_array.max())))
            print("ok", u, grp, flush=True)
        except Exception as e:  # noqa: BLE001
            fails.append(dict(StudyInstanceUID=u, error=str(e)[:150]))
            print("FALLO", u, e, flush=True)
    inten = pd.DataFrame(recs)
    inten.to_csv(OUT / "intensidades_corte_central.csv", index=False)
    print(f"Leídos: {len(inten)} | fallos: {len(fails)}")
    print(inten.groupby("grupo")[["hu_min", "hu_p1", "hu_p50", "hu_p99", "hu_max", "frac_aire", "frac_hueso"]]
          .median().round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    grupos = list(inten.grupo.unique())
    for g, d in inten.groupby("grupo"):
        for u in d.StudyInstanceUID:
            ax[0].hist(cache_hu[u].ravel(), bins=200, range=(-1100, 2000), histtype="step", alpha=.6,
                       label=g if u == d.StudyInstanceUID.iloc[0] else None, color=f"C{grupos.index(g)}")
    ax[0].set(yscale="log", title="Histograma de HU (corte central) por grupo de rescale", xlabel="HU", ylabel="píxeles (log)")
    ax[0].legend(fontsize=8)
    inten.boxplot(column="hu_p99", by="grupo", ax=ax[1], grid=False)
    ax[1].set(title="Percentil 99 de HU por grupo", xlabel="", ylabel="HU")
    ax[1].tick_params(axis="x", labelsize=7)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIG / "05_intensidades_hu.png", dpi=150, bbox_inches="tight")

    lab = labels.set_index("StudyInstanceUID").patient_overall
    sel = [u for u in cache_hu if lab.get(u) == 0][:2] + [u for u in cache_hu if lab.get(u) == 1][:2]
    sel += [u for u in atip if u in cache_hu and u not in sel][:1]   # sin repetir un estudio ya mostrado
    fig, ax = plt.subplots(len(sel), 2, figsize=(7, 3.4 * len(sel)))
    ax = np.atleast_2d(ax)
    for i, u in enumerate(sel):
        hu = cache_hu[u]
        lo, hi = np.percentile(hu, [0.5, 99.5])
        ax[i, 0].imshow(np.clip(hu, lo, hi), cmap="gray")
        ax[i, 0].set_title(f"Antes · HU recortado p0.5–p99.5\n…{u.split('.')[-1]} (y={lab.get(u)})", fontsize=8)
        ax[i, 1].imshow(_window_resize(hu), cmap="gray", vmin=0, vmax=1)
        ax[i, 1].set_title("Después · ventana ósea L400/W1800, 224×224", fontsize=8)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "06_antes_despues_preprocesamiento.png", dpi=150, bbox_inches="tight")
    print("Estudios mostrados en la figura 06:", sel)


if __name__ == "__main__":
    main()
