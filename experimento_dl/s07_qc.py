"""Control de calidad de los artefactos de entrenamiento/evaluación (no modifica resultados).

Comprueba: particiones disjuntas; NaN en historiales; métricas JSON == recalculadas a partir de
las predicciones guardadas; umbral == Youden en validación; checkpoints cargables; inferencia
desde DICOM crudo (inference.py, proceso aparte) == probabilidades de evaluación en estudios de
prueba no vistos. Escribe RESULTS_DIR/qc_report.json y termina con código 1 si algo falla.
Uso: python s07_qc.py [--n-infer 3]
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import DICOM_DIR, RESULTS_DIR, WEIGHTS_DIR
from s04_train_eval import metrics

INFER_SNIPPET = r"""
import sys, json, pathlib, pandas as pd
from inference import load_classifier, predict_study
ck, dicom, csv, n = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
clf = load_classifier(ck)
ref = pd.read_csv(csv).head(n)
out = []
for _, r in ref.iterrows():
    o = predict_study(sorted((dicom / r.StudyInstanceUID).glob("*.dcm")), clf)
    out.append(dict(uid=r.StudyInstanceUID, eval_prob=float(r.prob), infer_prob=o["probability"],
                    eval_pred=int(r.pred), infer_pred=o["prediction"]))
print("JSON:" + json.dumps(out))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-infer", type=int, default=3)
    ap.add_argument("--out", default="qc_report.json", help="nombre del informe en RESULTS_DIR")
    args = ap.parse_args()
    rep, ok = {}, True

    sp = pd.read_csv(RESULTS_DIR / "splits.csv")
    sets = {k: set(g) for k, g in sp.groupby("split")["StudyInstanceUID"]}
    disjoint = not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    rep["splits"] = dict(disjoint=disjoint, unique_ids=bool(sp.StudyInstanceUID.is_unique),
                         sizes={k: len(v) for k, v in sets.items()})
    ok &= disjoint and sp.StudyInstanceUID.is_unique

    for m in ("mil", "bigru"):
        mp = RESULTS_DIR / f"metrics_{m}.json"
        if not mp.exists():
            rep[m] = "sin entrenar"
            continue
        r = {}
        mt = json.loads(mp.read_text(encoding="utf-8"))
        h = pd.read_csv(RESULTS_DIR / f"history_{m}.csv")
        r["history_finite"] = bool(np.isfinite(h[["train_loss", "val_loss", "val_auc"]].values).all())
        va = pd.read_csv(RESULTS_DIR / f"val_predictions_{m}.csv")
        te = pd.read_csv(RESULTS_DIR / f"test_predictions_{m}.csv")
        r["pred_ids_in_split"] = bool(set(va.StudyInstanceUID) <= sets["val"] and set(te.StudyInstanceUID) <= sets["test"])
        fpr, tpr, thr = roc_curve(va.patient_overall, va.prob)
        thr_val = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
        r["threshold_matches_val_youden"] = bool(abs(thr_val - mt["val"]["threshold"]) < 1e-6)
        rec = metrics(te.patient_overall.values, te.prob.values, mt["test_thr_val_youden"]["threshold"])
        diffs = {k: abs(rec[k] - mt["test_thr_val_youden"][k]) for k in rec if isinstance(rec[k], float)}
        r["test_metrics_match_predictions"] = bool(max(diffs.values()) < 1e-6)
        r["n_test_evaluated"] = int(len(te))
        ck = WEIGHTS_DIR / f"{m}_final.pt"
        r["checkpoint_exists"] = ck.exists()
        if ck.exists() and args.n_infer and DICOM_DIR.exists():
            proc = subprocess.run([sys.executable, "-c", INFER_SNIPPET, str(ck), str(DICOM_DIR),
                                   str(RESULTS_DIR / f"test_predictions_{m}.csv"), str(args.n_infer)],
                                  capture_output=True, text=True,
                                  env={k: v for k, v in __import__("os").environ.items()
                                       if k not in ("RSNA_EXP", "K_SLICES", "CONTEXT_MM")})
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("JSON:")]
            if line:
                inf = json.loads(line[0][5:])
                md = max(abs(x["eval_prob"] - x["infer_prob"]) for x in inf)
                r["inference_unseen_test"] = dict(n=len(inf), max_abs_diff=md,
                                                  same_predictions=all(x["eval_pred"] == x["infer_pred"] for x in inf))
                r["inference_consistent"] = bool(md < 1e-3)
            else:
                r["inference_consistent"] = False
                r["inference_error"] = proc.stderr[-800:]
        checks = [v for k, v in r.items() if isinstance(v, bool)]
        ok &= all(checks)
        rep[m] = r

    rep["all_ok"] = bool(ok)
    (RESULTS_DIR / args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
