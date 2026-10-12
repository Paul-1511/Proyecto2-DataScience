"""Fase 6: comparación de modelos e informe a partir de resultados REALES.

Lee results/metrics_*.json y results/test_predictions_*.csv (generados por s04).
Si un modelo no existe, se omite; nunca se rellenan valores.
Genera: results/comparacion_modelos.csv, figuras comparativas y results/INFORME_RESULTADOS.md
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIG_DIR, K_SLICES, N_STUDIES, RESULTS_DIR  # noqa: E402

NAMES = {"mil": "ResNet18 + MIL (top-3)", "bigru": "ResNet18 + BiGRU"}
COLORS = {"mil": "#2a6fdb", "bigru": "#e07a1f"}
KEYS = ["roc_auc", "sensitivity", "specificity", "precision", "f1", "accuracy"]


def load():
    res = {}
    for m in NAMES:
        p = RESULTS_DIR / f"metrics_{m}.json"
        if p.exists():
            res[m] = dict(metrics=json.loads(p.read_text(encoding="utf-8")),
                          preds=pd.read_csv(RESULTS_DIR / f"test_predictions_{m}.csv"),
                          hist=pd.read_csv(RESULTS_DIR / f"history_{m}.csv"))
    return res


def table(res):
    rows = []
    for m, r in res.items():
        mt = r["metrics"]
        for part, key in [("validación", "val"), ("prueba (umbral val)", "test_thr_val_youden"),
                          ("prueba (umbral 0.5)", "test_thr_0_5")]:
            d = mt[key]
            rows.append(dict(modelo=NAMES[m], conjunto=part, n=d["n"], umbral=round(d["threshold"], 3),
                             **{k: round(d[k], 3) for k in KEYS},
                             TP=d["tp"], FP=d["fp"], TN=d["tn"], FN=d["fn"]))
        rows[-2]["AUC_IC95_bootstrap"] = "[%.3f, %.3f]" % tuple(mt["test_auc_ci95_bootstrap"])
        rows[-2]["tiempo_entrenamiento_s"] = mt["train_time_s"]
        rows[-2]["inferencia_ms_estudio"] = mt["inference_ms_per_study"]
        rows[-2]["mejor_epoca_val"] = mt["best_epoch"]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "comparacion_modelos.csv", index=False)
    return df


def figures(res):
    # 1) ROC de prueba superpuestas
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for m, r in res.items():
        p = r["preds"]
        fpr, tpr, _ = roc_curve(p["patient_overall"], p["prob"])
        auc = r["metrics"]["test_thr_val_youden"]["roc_auc"]
        lo, hi = r["metrics"]["test_auc_ci95_bootstrap"]
        ax.plot(fpr, tpr, color=COLORS[m], lw=2, label=f"{NAMES[m]}  AUC={auc:.3f} [{lo:.2f}-{hi:.2f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("sensibilidad")
    ax.set_title("ROC en prueba (estudios no vistos)"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "comparacion_roc_prueba.png", dpi=120); plt.close(fig)

    # 2) Barras de métricas en prueba (umbral elegido en validación)
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / max(len(res), 1)
    for i, (m, r) in enumerate(res.items()):
        vals = [r["metrics"]["test_thr_val_youden"][k] for k in KEYS]
        x = np.arange(len(KEYS)) + i * w
        ax.bar(x, vals, w, color=COLORS[m], label=NAMES[m])
        for xi, v in zip(x, vals):
            ax.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(KEYS)) + w * (len(res) - 1) / 2, KEYS)
    ax.set_ylim(0, 1.08); ax.set_title("Métricas en prueba (umbral fijado en validación)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_metricas_prueba.png", dpi=120); plt.close(fig)

    # 3) Distribución de probabilidades predichas por clase real
    fig, axes = plt.subplots(1, len(res), figsize=(4.5 * len(res), 3.8), squeeze=False)
    for ax, (m, r) in zip(axes[0], res.items()):
        p = r["preds"]
        for lab, c in [(0, "#888"), (1, "#c0392b")]:
            v = p.loc[p["patient_overall"] == lab, "prob"]
            ax.scatter(v, np.random.default_rng(0).uniform(-.2, .2, len(v)) + lab, color=c, alpha=.7,
                       label=f"real={lab} (n={len(v)})")
        ax.axvline(r["metrics"]["test_thr_val_youden"]["threshold"], color="k", ls="--", lw=1,
                   label="umbral (val)")
        ax.set_yticks([0, 1], ["sin fractura", "con fractura"]); ax.set_xlim(0, 1)
        ax.set_xlabel("probabilidad predicha"); ax.set_title(NAMES[m], fontsize=10)
        ax.legend(fontsize=7, loc="center right")
    fig.suptitle("Probabilidades por estudio en prueba"); fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_probabilidades_prueba.png", dpi=120); plt.close(fig)

    # 4) AUC de validación por época
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m, r in res.items():
        h = r["hist"]
        ax.plot(h["epoch"], h["val_auc"], marker="o", color=COLORS[m], label=NAMES[m])
    ax.set_xlabel("época"); ax.set_ylabel("ROC-AUC validación"); ax.legend(fontsize=8)
    ax.set_title("Selección de época (solo validación)"); fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_auc_validacion.png", dpi=120); plt.close(fig)

    # 5) Tiempos de entrenamiento e inferencia
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    ms = list(res)
    for a, key, title in [(ax[0], "train_time_s", "Entrenamiento total (s)"),
                          (ax[1], "inference_ms_per_study", "Inferencia (ms/estudio, GPU)")]:
        vals = [res[m]["metrics"][key] for m in ms]
        a.bar([NAMES[m] for m in ms], vals, color=[COLORS[m] for m in ms])
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        a.set_title(title, fontsize=10); a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "comparacion_tiempos.png", dpi=120); plt.close(fig)


def report(res, df):
    splits = pd.read_csv(RESULTS_DIR / "splits.csv")
    dl = pd.read_csv(RESULTS_DIR / "download_log.csv")
    audit = pd.read_csv(RESULTS_DIR / "orden_z_auditoria.csv")
    meta = pd.read_csv(RESULTS_DIR / "dicom_metadata.csv")
    used = set(audit["StudyInstanceUID"])
    s = splits[splits["StudyInstanceUID"].isin(used)].groupby("split")["patient_overall"].agg(["size", "sum"])

    L = ["# Resultados del experimento (generado automáticamente por s05_compare_report.py)", "",
         "Todos los valores provienen de `results/*.json|csv` producidos en esta ejecución.", "",
         "## Datos", "",
         f"- Muestra planificada: {N_STUDIES} estudios estratificados por `patient_overall` (semilla 42).",
         f"- Estudios descargados OK: {int((dl['status'] == 'ok').sum())}; preprocesados: {len(used)}.",
         f"- Cortes por estudio: {K_SLICES}, equiespaciados entre el 5% y 95% del rango de instancias.",
         f"- Volumen DICOM descargado: {dl['mb'].sum() / 1e3:.2f} GB.", "",
         "| partición | estudios | positivos |", "|---|---|---|"]
    L += [f"| {k} | {int(v['size'])} | {int(v['sum'])} |" for k, v in s.iterrows()]
    L += ["", "## Preprocesamiento (auditoría)", "",
          f"- Estudios con z únicos: {int(audit['z_unique'].sum())}/{len(audit)}; "
          f"z monótono respecto a la instancia: {int(audit['z_monotonic_vs_instance'].sum())}/{len(audit)}.",
          f"- Dirección z vs. instancia: {audit['z_direction_vs_instance'].value_counts().to_dict()}.",
          f"- RescaleSlope: {meta['slope'].value_counts().to_dict()}; "
          f"RescaleIntercept: {meta['intercept'].value_counts().to_dict()}.",
          f"- Transfer syntax: {meta['transfer_syntax'].value_counts().to_dict()}.", ""]
    if not res:
        L += ["## Modelos", "", "_Aún no hay modelos entrenados: no se reportan métricas._"]
    else:
        L += ["## Modelos (prueba: estudios nunca usados para entrenar ni para elegir época/umbral)", "",
              df.to_markdown(index=False), ""]
        if len(res) == 2:
            a, b = (res[m]["metrics"]["test_thr_val_youden"]["roc_auc"] for m in ("mil", "bigru"))
            ci = {m: res[m]["metrics"]["test_auc_ci95_bootstrap"] for m in res}
            overlap = ci["mil"][0] <= ci["bigru"][1] and ci["bigru"][0] <= ci["mil"][1]
            L += [f"AUC prueba: MIL={a:.3f}, BiGRU={b:.3f}. Los IC95% bootstrap "
                  f"{'se solapan' if overlap else 'no se solapan'}; "
                  + ("con este tamaño de prueba la diferencia no es concluyente." if overlap
                     else "aun así es una sola partición y una sola semilla."), ""]
    (RESULTS_DIR / "INFORME_RESULTADOS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    r = load()
    d = table(r) if r else pd.DataFrame()
    if r:
        figures(r)
    report(r, d)
    print(d.to_string() if len(d) else "Sin modelos entrenados todavía.")
