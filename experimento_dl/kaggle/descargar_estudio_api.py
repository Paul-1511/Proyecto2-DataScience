"""Descarga por la API de Kaggle todos los cortes DICOM de estudios concretos (uno a uno, con pausas
y reintentos ante HTTP 429). Alternativa a exportar_estudios_prueba.py cuando el notebook tarda.

    .venv/bin/python experimento_dl/kaggle/descargar_estudio_api.py <UID> [<UID> ...]

Los archivos de RSNA 2022 se llaman <InstanceNumber>.dcm (1..N); N se toma de
artifacts/v2/results/dicom_metadata.csv.gz. Destino: experimento_dl/data/dicom/<UID>/ (no versionado).
Requiere credenciales de Kaggle configuradas por el usuario y reglas de la competencia aceptadas.
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
COMP = "rsna-2022-cervical-spine-fracture-detection"


def main(uids):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    md = pd.read_csv(REPO / "artifacts/v2/results/dicom_metadata.csv.gz")
    n_files = md.groupby("StudyInstanceUID").n_files_study.first()
    for uid in uids:
        dest = ROOT / "data" / "dicom" / uid
        dest.mkdir(parents=True, exist_ok=True)
        n = int(n_files[uid])
        for i in range(1, n + 1):
            if (dest / f"{i}.dcm").exists():
                continue
            for intento in range(6):
                try:
                    api.competition_download_file(COMP, f"train_images/{uid}/{i}.dcm", path=str(dest), quiet=True)
                    break
                except Exception as e:  # noqa: BLE001
                    espera = 10 * 2 ** intento if "429" in str(e) else 3
                    print(f"{uid} {i}.dcm: {type(e).__name__} {str(e)[:60]} → reintento en {espera}s", flush=True)
                    time.sleep(espera)
            else:
                sys.exit(f"No se pudo descargar {uid}/{i}.dcm")
            time.sleep(0.3)
            if i % 25 == 0:
                print(f"{uid}: {i}/{n}", flush=True)
        print(f"LISTO {uid}: {len(list(dest.glob('*.dcm')))}/{n} DICOM", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
