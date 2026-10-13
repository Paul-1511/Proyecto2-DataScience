"""Fase 2: muestra estratificada por patient_overall y partición por estudio.

v1: 160 estudios -> results/splits.csv. v2 (RSNA_EXP=v2): todos -> results_v2/splits.csv.
No se sobrescribe la versión de otra configuración. Proporción final 64/16/20.

- Muestra de N_STUDIES estudios estratificada por patient_overall.
- 80% desarrollo / 20% prueba (estratificado, por estudio).
- Desarrollo -> 80% entrenamiento / 20% validación (estratificado, por estudio).
En RSNA 2022 cada StudyInstanceUID corresponde a un paciente distinto (no se
publica un ID de paciente aparte), por lo que separar por estudio separa pacientes.
Las etiquetas solo se usan como objetivo y para estratificar, nunca como entrada.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import N_STUDIES, RESULTS_DIR, SEED, TRAIN_CSV

df = pd.read_csv(TRAIN_CSV)
assert df["StudyInstanceUID"].is_unique

if N_STUDIES and N_STUDIES < len(df):
    sample, _ = train_test_split(df, train_size=N_STUDIES, stratify=df["patient_overall"],
                                 random_state=SEED)
else:  # N_STUDIES=0 (v2): todos los estudios de train.csv
    sample = df
dev, test = train_test_split(sample, test_size=0.20, stratify=sample["patient_overall"],
                             random_state=SEED)
train, val = train_test_split(dev, test_size=0.20, stratify=dev["patient_overall"],
                              random_state=SEED)

out = pd.concat([train.assign(split="train"), val.assign(split="val"),
                 test.assign(split="test")])[["StudyInstanceUID", "patient_overall", "split"]]
assert out["StudyInstanceUID"].is_unique  # sin fuga entre particiones
dst = RESULTS_DIR / "splits.csv"
if dst.exists() and not pd.read_csv(dst).sort_values("StudyInstanceUID").reset_index(drop=True).equals(
        out.sort_values("StudyInstanceUID").reset_index(drop=True)):
    raise SystemExit(f"{dst} existe y difiere: no se sobrescribe silenciosamente.")
out.to_csv(dst, index=False)

summary = out.groupby("split")["patient_overall"].agg(n="size", positivos="sum")
summary["prop_pos"] = (summary["positivos"] / summary["n"]).round(3)
print(summary)
print("prevalencia train.csv completo:", round(df["patient_overall"].mean(), 3))
summary.to_csv(RESULTS_DIR / "splits_summary.csv")
