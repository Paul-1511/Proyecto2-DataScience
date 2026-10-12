"""Figuras reales de preprocesamiento para la presentación (solo estudios descargados)."""
import ast

import matplotlib
import numpy as np
import pandas as pd
import pydicom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DICOM_DIR, FIG_DIR, RESULTS_DIR  # noqa: E402
from s03_preprocess import build_volume, slice_record  # noqa: E402

meta = pd.read_csv(RESULTS_DIR / "dicom_metadata.csv")
labels = pd.read_csv(RESULTS_DIR / "splits.csv").set_index("StudyInstanceUID")["patient_overall"]
uids = sorted(meta["StudyInstanceUID"].unique())

# 1) Antes / después en 6 estudios (corte central)
sel = uids[:6]
fig, ax = plt.subplots(2, len(sel), figsize=(2.6 * len(sel), 5.6))
for j, uid in enumerate(sel):
    files = sorted((DICOM_DIR / uid).glob("*.dcm"), key=lambda p: int(p.stem))
    recs = [slice_record(pydicom.dcmread(f), f.name, uid, int(f.stem)) for f in files]
    vol, recs = build_volume(recs)
    k = len(recs) // 2
    ax[0, j].imshow(np.clip(recs[k]["hu"], -1024, 2000), cmap="gray")
    ax[0, j].set_title(f"…{uid[-5:]} · y={labels[uid]}", fontsize=8)
    ax[1, j].imshow(vol[k], cmap="gray", vmin=0, vmax=255)
    for a in ax[:, j]:
        a.axis("off")
ax[0, 0].text(-0.15, 0.5, "ANTES\nHU crudo", transform=ax[0, 0].transAxes, ha="right", va="center", fontsize=9)
ax[1, 0].text(-0.15, 0.5, "DESPUÉS\nventana ósea\n224×224", transform=ax[1, 0].transAxes, ha="right", va="center", fontsize=9)
fig.suptitle("Preprocesamiento DICOM real en estudios descargados (corte central, orden z)")
fig.tight_layout()
fig.savefig(FIG_DIR / "presentacion_antes_despues_multiestudio.png", dpi=120)
plt.close(fig)

# 2) Variabilidad de metadatos por estudio
st = meta.groupby("StudyInstanceUID").agg(
    grosor=("slice_thickness", "first"), spacing=("pixel_spacing", "first"),
    intercept=("intercept", "first"), sintaxis=("transfer_syntax", "first"))
st["pixel_mm"] = st["spacing"].map(lambda s: ast.literal_eval(s)[0] if isinstance(s, str) and s.startswith("[") else np.nan)
st["sintaxis"] = st["sintaxis"].str.replace(r"JPEG Lossless.*", "JPEG Lossless", regex=True)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].hist(st["grosor"].astype(float), bins=10, color="#2a6fdb"); ax[0].set_title("SliceThickness (mm)")
ax[1].hist(st["pixel_mm"], bins=10, color="#2a6fdb"); ax[1].set_title("PixelSpacing (mm)")
vc = st["sintaxis"].value_counts()
ax[2].barh(vc.index, vc.values, color="#e07a1f"); ax[2].set_title("Codificación (Transfer Syntax)")
for a in ax[:2]:
    a.set_ylabel("estudios")
fig.suptitle(f"Variabilidad de metadatos DICOM en {len(st)} estudios descargados")
fig.tight_layout()
fig.savefig(FIG_DIR / "presentacion_variabilidad_metadatos.png", dpi=120)
plt.close(fig)
st.to_csv(RESULTS_DIR / "metadatos_por_estudio.csv")
print(st[["grosor", "pixel_mm", "intercept", "sintaxis"]].to_string())
