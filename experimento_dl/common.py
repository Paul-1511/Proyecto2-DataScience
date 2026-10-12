"""Configuración compartida del experimento RSNA 2022.

Dos configuraciones experimentales (seleccionadas con la variable de entorno RSNA_EXP):
  v1 (por defecto): muestra de 160 estudios, 24 cortes 2D por estudio (canal replicado ×3).
                    Es la configuración usada por la app de demostración; no se modifica.
  v2              : todos los estudios de train.csv (2019), 48 posiciones por estudio en 2.5D
                    (vecinos a ±CONTEXT_MM como canales R/B). Pensada para Kaggle Notebooks.
Las rutas pueden redirigirse con RSNA_DICOM_DIR, RSNA_CACHE_DIR, RSNA_WEIGHTS_DIR y
RSNA_RESULTS_DIR (p. ej. a /kaggle/input y /kaggle/working).
"""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TRAIN_CSV = Path(os.environ.get("RSNA_TRAIN_CSV", REPO / "train.csv"))

EXP = os.environ.get("RSNA_EXP", "v1")
_CFG = {
    "v1": dict(n_studies=160, k_slices=24, context_mm=0.0, results="results", cache="cache"),
    "v2": dict(n_studies=0, k_slices=48, context_mm=2.0, results="results_v2", cache="cache_v2"),
}[EXP]

DATA_DIR = ROOT / "data"            # NO versionado (.gitignore)
# DICOM: descargados localmente (v1) o dataset montado en Kaggle:
#   RSNA_DICOM_DIR=/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images
DICOM_DIR = Path(os.environ.get("RSNA_DICOM_DIR", DATA_DIR / "dicom"))
CACHE_DIR = Path(os.environ.get("RSNA_CACHE_DIR", DATA_DIR / _CFG["cache"]))   # tensores preprocesados
WEIGHTS_DIR = Path(os.environ.get("RSNA_WEIGHTS_DIR", DATA_DIR / "weights"))   # pesos (no versionados)
RESULTS_DIR = Path(os.environ.get("RSNA_RESULTS_DIR", ROOT / _CFG["results"]))  # métricas y figuras
FIG_DIR = RESULTS_DIR / "figures"

COMPETITION = "rsna-2022-cervical-spine-fracture-detection"

SEED = 42
# N_STUDIES = 0 significa "todos los estudios de train.csv"
N_STUDIES = int(os.environ.get("N_STUDIES", _CFG["n_studies"]))
K_SLICES = int(os.environ.get("K_SLICES", _CFG["k_slices"]))        # posiciones fijas por estudio
CONTEXT_MM = float(os.environ.get("CONTEXT_MM", _CFG["context_mm"]))  # 0 = 2D; >0 = 2.5D
SLICE_MARGIN = 0.05                                                  # se descarta 5% en cada extremo
IMG_SIZE = 224

# Ventana ósea (HU). Nivel 400, ancho 1800 -> rango [-500, 1300].
BONE_WL, BONE_WW = 400, 1800


def preprocessing_config():
    """Configuración de preprocesamiento que se guarda junto a cada checkpoint."""
    return dict(exp=EXP, k_slices=K_SLICES, context_mm=CONTEXT_MM, slice_margin=SLICE_MARGIN,
                img_size=IMG_SIZE, bone_wl=BONE_WL, bone_ww=BONE_WW,
                order="ImagePositionPatient[2] ascendente",
                normalization="HU -> ventana ósea -> [0,1]; en la red: media/desv. ImageNet")


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_slice_numbers(n_slices: int, k: int = None, margin: float = SLICE_MARGIN):
    """Regla reproducible: k números de corte equiespaciados entre el 5% y el 95%
    del rango [1, n_slices] (nombres de archivo de Kaggle = número de instancia).
    NO garantiza que se incluyan los cortes donde la fractura es visible."""
    k = K_SLICES if k is None else k
    lo = 1 + margin * (n_slices - 1)
    hi = 1 + (1 - margin) * (n_slices - 1)
    idx = np.round(np.linspace(lo, hi, k)).astype(int)
    return sorted(set(int(i) for i in idx))


for d in (CACHE_DIR, WEIGHTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

if "RSNA_DICOM_DIR" not in os.environ:
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
