"""Verificación independiente de los artefactos y comparación pareada de modelos (sin reentrenar).

1) Recalcula desde los CSV de predicciones todas las métricas de metrics_<m>.json (validación,
   prueba con umbral de validación y con 0.5, IC95 % bootstrap del AUC) y las compara.
2) Comprueba: IDs de predicciones == IDs de cada partición; época elegida == argmax del AUC de
   validación en el historial; comparacion_modelos.csv == JSON.
3) Comparación pareada en el MISMO conjunto de prueba (análisis de evaluación, no de selección):
   diferencia de AUC (BiGRU - MIL) con bootstrap pareado y prueba de DeLong; McNemar exacto sobre
   las predicciones binarias con los umbrales de validación. Se repite en validación como referencia.
Salidas: RESULTS_DIR/verificacion_metricas.json y RESULTS_DIR/comparacion_pareada.json.
Uso: RSNA_EXP=v2 RSNA_RESULTS_DIR=<carpeta> python s08_verificacion.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import RESULTS_DIR, SEED
from s04_train_eval import bootstrap_auc, metrics

MODELS = ("mil", "bigru")
# Los CSV guardan probabilidades float32 con 8 dígitos significativos; al releerlas en float64 un
# estudio con probabilidad EXACTAMENTE igual al umbral (el umbral de Youden es la probabilidad de un
# estudio de validación) puede quedar ~1e-8 por debajo. Se recalcula con esta tolerancia.
THR_EPS = 1e-6


def close(a, b, tol=1e-6):
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return abs(a - b) <= tol


def delong(y, p1, p2):
    """Prueba de DeLong para dos AUC correlacionados (DeLong et al., 1988)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0

    def components(p):
        x, z = p[pos], p[neg]
        psi = (x[:, None] > z[None, :]).astype(float) + 0.5 * (x[:, None] == z[None, :])
        return psi.mean(1), psi.mean(0), psi.mean()

    v10a, v01a, auc_a = components(np.asarray(p1))
    v10b, v01b, auc_b = components(np.asarray(p2))
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / pos.sum() + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / neg.sum()
    zval = (auc_b - auc_a) / np.sqrt(var)
    return dict(auc_a=float(auc_a), auc_b=float(auc_b), diff=float(auc_b - auc_a), z=float(zval),
                p_value=float(2 * stats.norm.sf(abs(zval))))


def paired_bootstrap(y, p1, p2, n=2000, seed=SEED):
    rng, d = np.random.default_rng(seed), []
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    d = np.array(d)
    return dict(diff_mean=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                prop_diff_le_0=float((d <= 0).mean()), n_resamples=int(len(d)))


def mcnemar(y, a, b):
    ca, cb = (a == y), (b == y)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())   # solo A acierta / solo B acierta
    p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return dict(solo_mil_acierta=n01, solo_bigru_acierta=n10, p_value_exacto=p)


def main():
    sp = pd.read_csv(RESULTS_DIR / "splits.csv")
    ids = {k: set(g) for k, g in sp.groupby("split")["StudyInstanceUID"]}
    ver, ok = {}, True
    preds = {}
    for m in MODELS:
        mt = json.loads((RESULTS_DIR / f"metrics_{m}.json").read_text(encoding="utf-8"))
        va = pd.read_csv(RESULTS_DIR / f"val_predictions_{m}.csv")
        te = pd.read_csv(RESULTS_DIR / f"test_predictions_{m}.csv")
        h = pd.read_csv(RESULTS_DIR / f"history_{m}.csv")
        preds[m] = (va, te, mt)
        thr = mt["val"]["threshold"]
        chk = {}
        chk["ids_val_exactos"] = set(va.StudyInstanceUID) == ids["val"]
        chk["ids_test_exactos"] = set(te.StudyInstanceUID) == ids["test"]
        chk["sin_fuga_train"] = not ((set(va.StudyInstanceUID) | set(te.StudyInstanceUID)) & ids["train"])
        chk["etiquetas_coinciden_splits"] = bool(
            (te.set_index("StudyInstanceUID").patient_overall
             == sp.set_index("StudyInstanceUID").loc[te.StudyInstanceUID, "patient_overall"]).all())
        chk["pred_csv_coincide_umbral"] = bool((te.pred == (te.prob >= thr - THR_EPS).astype(int)).all()
                                               and (va.pred == (va.prob >= thr - THR_EPS).astype(int)).all())
        chk["estudios_en_el_limite_val_test"] = [int((abs(va.prob - thr) < THR_EPS).sum()),
                                                 int((abs(te.prob - thr) < THR_EPS).sum())]
        chk["probs_en_0_1_y_finitas"] = bool(np.isfinite(te.prob).all() and te.prob.between(0, 1).all())
        for key, frame, t in [("val", va, thr), ("test_thr_val_youden", te, thr), ("test_thr_0_5", te, 0.5)]:
            rec = metrics(frame.patient_overall.values, frame.prob.values, t - THR_EPS)
            chk[f"{key}_coincide"] = all(close(rec[k], mt[key][k]) for k in rec if k != "threshold")
        ci = bootstrap_auc(te.patient_overall.values, te.prob.values)
        chk["ic95_auc_reproducible"] = all(close(a, b, 1e-9) for a, b in zip(ci, mt["test_auc_ci95_bootstrap"]))
        chk["mejor_epoca_es_max_val_auc"] = int(h.loc[h.val_auc.idxmax(), "epoch"]) == mt["best_epoch"]
        chk["historial_finito"] = bool(np.isfinite(h[["train_loss", "val_loss", "val_auc"]].values).all())
        ok &= all(v for v in chk.values() if isinstance(v, bool))
        ver[m] = chk

    comp = pd.read_csv(RESULTS_DIR / "comparacion_modelos.csv")
    names = {"mil": "ResNet18 + MIL (top-3)", "bigru": "ResNet18 + BiGRU"}
    cc = True
    for m in MODELS:
        row = comp[(comp.modelo == names[m]) & (comp.conjunto == "prueba (umbral val)")].iloc[0]
        t = preds[m][2]["test_thr_val_youden"]
        cc &= all(close(float(row[k]), round(t[k], 3), 1e-9) for k in ("roc_auc", "sensitivity", "specificity", "f1"))
    ver["comparacion_csv_coincide_json"] = bool(cc)
    ok &= cc
    ver["all_ok"] = bool(ok)
    (RESULTS_DIR / "verificacion_metricas.json").write_text(json.dumps(ver, indent=2, ensure_ascii=False), encoding="utf-8")

    pair = {}
    for split, idx in (("test", 1), ("val", 0)):
        a = preds["mil"][idx].set_index("StudyInstanceUID")
        b = preds["bigru"][idx].set_index("StudyInstanceUID").loc[a.index]
        y = a.patient_overall.values
        pair[split] = dict(n=int(len(y)), positivos=int(y.sum()),
                           delong=delong(y, a.prob.values, b.prob.values),
                           bootstrap_pareado=paired_bootstrap(y, a.prob.values, b.prob.values),
                           mcnemar_umbral_val=mcnemar(y, a.pred.values, b.pred.values),
                           correlacion_probs_spearman=float(stats.spearmanr(a.prob, b.prob).statistic))
    pair["nota"] = ("Diferencias = BiGRU - MIL. Evaluación post hoc sobre el mismo conjunto; la selección de "
                    "modelo se definió a priori por AUC/IC y no se ajustó nada con la prueba.")
    (RESULTS_DIR / "comparacion_pareada.json").write_text(json.dumps(pair, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(verificacion=ver, comparacion_pareada=pair), indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
