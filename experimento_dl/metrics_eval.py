"""Módulo independiente de evaluación experimental — P3 (Hansel López).

Implementación independiente de métricas de clasificación para el experimento RSNA 2022.
No depende de s04_train_eval.py; reimplementa todas las funciones desde cero para
auditoría cruzada de los resultados reportados en artifacts/v2/results/.

Referencias:
- DeLong et al. (1988). Comparing the areas under two or more correlated ROC curves.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
- Youden (1950). Index for rating diagnostic tests. Cancer, 3(1), 32–35.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SEED = 42


# ─────────────────────────────────────────────────────────────
# 1. MÉTRICAS DE CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────

def confusion_matrix_detailed(y_true, y_pred):
    """Matriz de confusión con etiquetas legibles.

    Returns
    -------
    dict con 'matrix' (ndarray 2×2), 'labels', 'tn', 'fp', 'fn', 'tp'.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "matrix": cm,
        "labels": ["sin_fractura (0)", "con_fractura (1)"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_complete(y_true, y_prob, threshold=0.5):
    """Métricas completas de clasificación en un punto de umbral.

    Calcula: accuracy, precision, sensitivity (recall), specificity,
    F1, ROC-AUC, y matriz de confusión.

    Parameters
    ----------
    y_true : array-like de 0/1
    y_prob : array-like de probabilidades [0, 1]
    threshold : float, umbral de decisión

    Returns
    -------
    dict con todas las métricas
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix_detailed(y_true, y_pred)
    n = len(y_true)

    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = cm["tn"] / (cm["tn"] + cm["fp"]) if (cm["tn"] + cm["fp"]) > 0 else float("nan")
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    if len(set(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = float("nan")

    return {
        "threshold": float(threshold),
        "n": int(n),
        "tp": cm["tp"],
        "fp": cm["fp"],
        "tn": cm["tn"],
        "fn": cm["fn"],
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
    }


# ─────────────────────────────────────────────────────────────
# 2. SELECCIÓN DE UMBRAL
# ─────────────────────────────────────────────────────────────

def find_threshold_youden(y_true, y_prob):
    """Encuentra el umbral que maximiza el índice de Youden J = sens + spec - 1.

    Parameters
    ----------
    y_true : array-like de 0/1
    y_prob : array-like de probabilidades [0, 1]

    Returns
    -------
    float : umbral óptimo
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────
# 3. INTERVALOS DE CONFIANZA (BOOTSTRAP)
# ─────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_prob, n_resamples=2000, seed=SEED):
    """IC95% del ROC-AUC mediante bootstrap percentil.

    Parameters
    ----------
    y_true : array-like de 0/1
    y_prob : array-like de probabilidades [0, 1]
    n_resamples : int, número de remuestreos
    seed : int, semilla para reproducibilidad

    Returns
    -------
    list [lo, hi] : límites del IC95%
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = []

    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_boot = y_true[idx]
        p_boot = y_prob[idx]
        if len(set(y_boot)) == 2:
            aucs.append(roc_auc_score(y_boot, p_boot))

    if len(aucs) == 0:
        return [float("nan"), float("nan")]

    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def bootstrap_metric_ci(y_true, y_prob, threshold, metric_fn, n_resamples=2000,
                        seed=SEED):
    """IC95% bootstrap de una métrica arbitraria.

    Parameters
    ----------
    metric_fn : callable(y_true, y_prob, threshold) -> float
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_boot = y_true[idx]
        p_boot = y_prob[idx]
        if len(set(y_boot)) == 2:
            values.append(metric_fn(y_boot, p_boot, threshold))

    if len(values) == 0:
        return [float("nan"), float("nan")]

    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# ─────────────────────────────────────────────────────────────
# 4. PRUEBAS ESTADÍSTICAS PAREADAS
# ─────────────────────────────────────────────────────────────

def delong_test(y_true, prob_a, prob_b):
    """Prueba de DeLong para comparar dos AUC correlacionados.

    Implementación basada en el método estructural de DeLong et al. (1988).
    H₀: AUC_A = AUC_B  vs  H₁: AUC_A ≠ AUC_B

    Parameters
    ----------
    y_true : array-like de 0/1
    prob_a, prob_b : array-like de probabilidades de los dos modelos

    Returns
    -------
    dict con auc_a, auc_b, diff, z_stat, p_value (bilateral)
    """
    y = np.asarray(y_true, dtype=int)
    p1 = np.asarray(prob_a, dtype=float)
    p2 = np.asarray(prob_b, dtype=float)

    pos, neg = y == 1, y == 0

    def _placement_values(probs):
        """Calcula los valores de colocación (V10 y V01) para positivos y negativos."""
        x_pos = probs[pos]
        x_neg = probs[neg]
        # psi[i,j] = I(x_pos[i] > x_neg[j]) + 0.5 * I(x_pos[i] == x_neg[j])
        psi = (x_pos[:, None] > x_neg[None, :]).astype(float) + \
              0.5 * (x_pos[:, None] == x_neg[None, :]).astype(float)
        v10 = psi.mean(axis=1)  # para cada positivo
        v01 = psi.mean(axis=0)  # para cada negativo
        auc_val = psi.mean()
        return v10, v01, auc_val

    v10_a, v01_a, auc_a = _placement_values(p1)
    v10_b, v01_b, auc_b = _placement_values(p2)

    # Covarianzas
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))

    n_pos = pos.sum()
    n_neg = neg.sum()

    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n_pos + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_neg

    diff = auc_b - auc_a
    z_stat = diff / np.sqrt(var) if var > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z_stat))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "diff": float(diff),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


def paired_bootstrap_test(y_true, prob_a, prob_b, n_resamples=2000, seed=SEED):
    """Bootstrap pareado para la diferencia de AUC entre dos modelos.

    Parameters
    ----------
    y_true : array-like de 0/1
    prob_a, prob_b : probabilidades de los dos modelos
    n_resamples : int
    seed : int

    Returns
    -------
    dict con diff_mean, ci95, prop_diff_le_0, n_resamples
    """
    y = np.asarray(y_true, dtype=int)
    p1 = np.asarray(prob_a, dtype=float)
    p2 = np.asarray(prob_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        y_boot = y[idx]
        if len(set(y_boot)) < 2:
            continue
        auc_a_boot = roc_auc_score(y_boot, p1[idx])
        auc_b_boot = roc_auc_score(y_boot, p2[idx])
        diffs.append(auc_b_boot - auc_a_boot)

    diffs = np.array(diffs)
    return {
        "diff_mean": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "prop_diff_le_0": float((diffs <= 0).mean()),
        "n_valid_resamples": int(len(diffs)),
    }


def mcnemar_exact_test(y_true, pred_a, pred_b):
    """Prueba exacta de McNemar para comparar dos clasificadores binarios.

    Evalúa si la diferencia en el número de aciertos exclusivos es significativa.
    Usa la distribución binomial B(n01+n10, 0.5) bajo H₀.

    Parameters
    ----------
    y_true : array-like de 0/1
    pred_a, pred_b : predicciones binarias de los dos modelos

    Returns
    -------
    dict con: solo_a_acierta, solo_b_acierta, ambos_aciertan, ambos_fallan, p_value
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)

    correct_a = (a == y)
    correct_b = (b == y)

    n01 = int((correct_a & ~correct_b).sum())  # solo A acierta
    n10 = int((~correct_a & correct_b).sum())  # solo B acierta
    n11 = int((correct_a & correct_b).sum())    # ambos aciertan
    n00 = int((~correct_a & ~correct_b).sum())  # ambos fallan

    total_disc = n01 + n10
    if total_disc == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(min(n01, n10), total_disc, 0.5).pvalue)

    return {
        "solo_a_acierta": n01,
        "solo_b_acierta": n10,
        "ambos_aciertan": n11,
        "ambos_fallan": n00,
        "p_value": float(p_value),
    }


def spearman_correlation(prob_a, prob_b):
    """Correlación de Spearman entre las probabilidades de dos modelos."""
    return float(stats.spearmanr(prob_a, prob_b).statistic)


# ─────────────────────────────────────────────────────────────
# 5. VALIDACIÓN DE ARCHIVOS
# ─────────────────────────────────────────────────────────────

REQUIRED_PRED_COLUMNS = {"StudyInstanceUID", "patient_overall", "split", "prob", "pred"}
REQUIRED_SPLIT_COLUMNS = {"StudyInstanceUID", "patient_overall", "split"}
VALID_SPLITS = {"train", "val", "test"}


def verify_predictions_format(csv_path):
    """Verifica que un CSV de predicciones tiene el formato esperado.

    Returns
    -------
    dict con 'valid' (bool), 'n_rows', 'columns', 'issues' (lista de str).
    """
    path = Path(csv_path)
    issues = []

    if not path.exists():
        return {"valid": False, "n_rows": 0, "columns": [],
                "issues": [f"Archivo no encontrado: {path}"]}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"valid": False, "n_rows": 0, "columns": [],
                "issues": [f"Error al leer CSV: {e}"]}

    cols = set(df.columns)
    missing = REQUIRED_PRED_COLUMNS - cols
    if missing:
        issues.append(f"Columnas faltantes: {missing}")

    if "prob" in cols:
        if not df["prob"].between(0, 1).all():
            issues.append("Hay probabilidades fuera de [0, 1]")
        if not np.isfinite(df["prob"]).all():
            issues.append("Hay probabilidades no finitas")

    if "pred" in cols:
        if not df["pred"].isin([0, 1]).all():
            issues.append("Las predicciones binarias no son solo 0/1")

    if "split" in cols:
        invalid_splits = set(df["split"].unique()) - VALID_SPLITS
        if invalid_splits:
            issues.append(f"Splits inválidos: {invalid_splits}")

    return {
        "valid": len(issues) == 0,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "issues": issues,
    }


def verify_partitions_disjoint(splits_csv):
    """Verifica que las particiones train/val/test son disjuntas por StudyInstanceUID.

    Returns
    -------
    dict con 'disjoint' (bool), 'sizes', 'issues', 'uids_per_split'.
    """
    path = Path(splits_csv)
    issues = []

    if not path.exists():
        return {"disjoint": False, "sizes": {}, "issues": [f"Archivo no encontrado: {path}"],
                "uids_per_split": {}}

    df = pd.read_csv(path)
    missing = REQUIRED_SPLIT_COLUMNS - set(df.columns)
    if missing:
        issues.append(f"Columnas faltantes: {missing}")
        return {"disjoint": False, "sizes": {}, "issues": issues, "uids_per_split": {}}

    uids_per_split = {}
    for split, grp in df.groupby("split"):
        uids = set(grp["StudyInstanceUID"])
        uids_per_split[split] = len(uids)
        # Verificar unicidad dentro del split
        if len(uids) != len(grp):
            issues.append(f"UIDs duplicados en split '{split}'")

    # Verificar disjunción
    all_uids = []
    for split, grp in df.groupby("split"):
        all_uids.extend(grp["StudyInstanceUID"].tolist())
    if len(all_uids) != len(set(all_uids)):
        issues.append("Hay UIDs compartidos entre particiones (no son disjuntas)")

    sizes = {k: int(v) for k, v in df["split"].value_counts().to_dict().items()}

    return {
        "disjoint": len(issues) == 0,
        "sizes": sizes,
        "issues": issues,
        "uids_per_split": uids_per_split,
    }


def verify_metrics_consistency(metrics_json, predictions_csv, threshold_key="val"):
    """Verifica que las métricas del JSON coinciden con las predicciones del CSV.

    Recalcula las métricas desde las predicciones y las compara con las reportadas.

    Parameters
    ----------
    metrics_json : ruta al JSON de métricas
    predictions_csv : ruta al CSV de predicciones
    threshold_key : 'val' para usar umbral de validación

    Returns
    -------
    dict con 'consistent' (bool), 'max_diff', 'details'
    """
    mt = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    df = pd.read_csv(predictions_csv)
    THR_EPS = 1e-6

    y = df["patient_overall"].values
    p = df["prob"].values

    issues = []
    max_diff = 0.0

    # Verificar umbral de validación
    thr_val = mt["val"]["threshold"]
    rec = metrics_complete(y, p, thr_val - THR_EPS)

    for key in ["sensitivity", "specificity", "precision", "f1", "accuracy", "roc_auc",
                "tp", "fp", "tn", "fn"]:
        reported = mt["test_thr_val_youden"][key]
        computed = rec[key]
        diff = abs(reported - computed)
        max_diff = max(max_diff, diff)
        if diff > 1e-4:
            issues.append(f"{key}: reportado={reported}, recalculado={computed}, diff={diff:.2e}")

    # Verificar IC95% bootstrap
    ci_reported = mt["test_auc_ci95_bootstrap"]
    ci_computed = bootstrap_auc_ci(y, p)
    for i, (lo, hi) in enumerate(zip(ci_reported, ci_computed)):
        diff = abs(lo - hi)
        max_diff = max(max_diff, diff)
        if diff > 1e-4:
            issues.append(f"IC95%[{i}]: reportado={lo}, recalculado={hi}")

    return {
        "consistent": len(issues) == 0,
        "max_diff": float(max_diff),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────
# 6. ANÁLISIS DE ERRORES
# ─────────────────────────────────────────────────────────────

def error_analysis(y_true, prob, threshold, uids=None):
    """Análisis de errores por tipo: FP, FN, TP, TN.

    Parameters
    ----------
    y_true : array-like de 0/1
    prob : array-like de probabilidades
    threshold : float
    uids : array-like opcional de identificadores

    Returns
    -------
    dict con dataframes para cada tipo de error y estadísticas resumen
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)

    tp_mask = (y == 1) & (pred == 1)
    fp_mask = (y == 0) & (pred == 1)
    fn_mask = (y == 1) & (pred == 0)
    tn_mask = (y == 0) & (pred == 0)

    result = {
        "summary": {
            "n_total": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "tp": int(tp_mask.sum()),
            "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()),
            "tn": int(tn_mask.sum()),
            "threshold": float(threshold),
        },
        "fp_indices": np.where(fp_mask)[0].tolist(),
        "fn_indices": np.where(fn_mask)[0].tolist(),
        "tp_indices": np.where(tp_mask)[0].tolist(),
        "tn_indices": np.where(tn_mask)[0].tolist(),
        "fp_probs": p[fp_mask].tolist(),
        "fn_probs": p[fn_mask].tolist(),
    }

    if uids is not None:
        uids = np.asarray(uids)
        result["fp_uids"] = uids[fp_mask].tolist()
        result["fn_uids"] = uids[fn_mask].tolist()

    return result


def disagreement_analysis(y_true, prob_a, prob_b, threshold_a, threshold_b,
                          uids=None):
    """Análisis de desacuerdos entre dos modelos.

    Returns
    -------
    dict con casos donde los modelos discrepan entre sí.
    """
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    pred_a = (pa >= threshold_a).astype(int)
    pred_b = (pb >= threshold_b).astype(int)

    # Solo A acierta, solo B acierta, ambos aciertan, ambos fallan
    only_a = (pred_a == y) & (pred_b != y)
    only_b = (pred_b == y) & (pred_a != y)
    both_correct = (pred_a == y) & (pred_b == y)
    both_wrong = (pred_a != y) & (pred_b != y)

    result = {
        "summary": {
            "n_total": int(len(y)),
            "solo_a_acierta": int(only_a.sum()),
            "solo_b_acierta": int(only_b.sum()),
            "ambos_aciertan": int(both_correct.sum()),
            "ambos_fallan": int(both_wrong.sum()),
        },
        "only_a_correct_indices": np.where(only_a)[0].tolist(),
        "only_b_correct_indices": np.where(only_b)[0].tolist(),
        "both_wrong_indices": np.where(both_wrong)[0].tolist(),
    }

    if uids is not None:
        uids = np.asarray(uids)
        result["only_a_correct_uids"] = uids[only_a].tolist()
        result["only_b_correct_uids"] = uids[only_b].tolist()
        result["both_wrong_uids"] = uids[both_wrong].tolist()

    return result
