"""Orquestador: preprocesa lo descargado y entrena/evalúa ambos modelos con las particiones fijas.

Solo usa estudios descargados y preprocesados; las particiones (results/splits.csv) no se modifican.
Registra en results/particiones_efectivas.csv cuántos estudios de cada partición se usaron.
Uso: python run_experiment.py [--epochs 12]
"""
import argparse
import subprocess
import sys

import pandas as pd

from common import CACHE_DIR, RESULTS_DIR


def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    sh("s03_preprocess.py")
    sp = pd.read_csv(RESULTS_DIR / "splits.csv")
    sp["disponible"] = sp["StudyInstanceUID"].map(lambda u: (CACHE_DIR / f"{u}.npy").exists())
    eff = sp.groupby("split").agg(planificados=("StudyInstanceUID", "size"),
                                  usados=("disponible", "sum"),
                                  positivos_usados=("patient_overall", lambda s: int(s[sp.loc[s.index, "disponible"]].sum())))
    eff.to_csv(RESULTS_DIR / "particiones_efectivas.csv")
    print(eff, flush=True)
    for name in ("val", "test"):
        u = eff.loc[name]
        if u["usados"] < 4 or u["positivos_usados"] in (0, u["usados"]):
            raise SystemExit(f"Partición {name} insuficiente para evaluar (requiere ambas clases).")

    for m in ("mil", "bigru"):
        sh("s04_train_eval.py", "--model", m, "--epochs", str(args.epochs))
    sh("s05_compare_report.py")


if __name__ == "__main__":
    main()
