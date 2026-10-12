"""Configuración compartida del experimento RSNA 2022 (muestra reducida)."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TRAIN_CSV = REPO / "train.csv"

DATA_DIR = ROOT / "data"            # NO versionado (.gitignore)
# DICOM descargados. En Kaggle Notebooks puede apuntarse al dataset montado completo:
#   RSNA_DICOM_DIR=/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images
DICOM_DIR = Path(os.environ.get("RSNA_DICOM_DIR", DATA_DIR / "dicom"))
CACHE_DIR = DATA_DIR / "cache"      # tensores preprocesados (.npy)
WEIGHTS_DIR = DATA_DIR / "weights"  # pesos (no versionados)
RESULTS_DIR = ROOT / "results"      # métricas y figuras (versionado)
FIG_DIR = RESULTS_DIR / "figures"

COMPETITION = "rsna-2022-cervical-spine-fracture-detection"

SEED = 42
N_STUDIES = int(os.environ.get("N_STUDIES", 160))   # tamaño de la muestra
K_SLICES = int(os.environ.get("K_SLICES", 24))      # cortes fijos por estudio
SLICE_MARGIN = 0.05                                  # se descarta 5% en cada extremo
IMG_SIZE = 224

# Ventana ósea (HU). Nivel 400, ancho 1800 -> rango [-500, 1300].
BONE_WL, BONE_WW = 400, 1800


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_slice_numbers(n_slices: int, k: int = K_SLICES, margin: float = SLICE_MARGIN):
    """Regla reproducible: k números de corte equiespaciados entre el 5% y el 95%
    del rango [1, n_slices] (nombres de archivo de Kaggle = número de instancia).
    NO garantiza que se incluyan los cortes donde la fractura es visible."""
    lo = 1 + margin * (n_slices - 1)
    hi = 1 + (1 - margin) * (n_slices - 1)
    idx = np.round(np.linspace(lo, hi, k)).astype(int)
    return sorted(set(int(i) for i in idx))


for d in (CACHE_DIR, WEIGHTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

if "RSNA_DICOM_DIR" not in os.environ:
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
