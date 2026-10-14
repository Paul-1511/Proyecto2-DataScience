from pathlib import Path

import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# 04 · Integración de la aplicación Streamlit — RSNA 2022

**Responsable:** Cabrera (P1 · aplicación e inferencia) · rama `feature/streamlit`

Este notebook demuestra, sin implementar ningún modelo nuevo, cómo la app:
1. carga el módulo de inferencia (`experimento_dl/inference.py`);
2. verifica la disponibilidad e integridad de los checkpoints;
3. prepara un estudio DICOM;
4. obtiene una predicción real **si existe un checkpoint**;
5. comprueba la compatibilidad del preprocesamiento con el entrenamiento.

Toda la lógica vive en `experimento_dl/app_core.py` (reutilizada por la app y las pruebas); aquí solo se invoca."""),
md("""## DEPENDENCIAS E INTEGRACIÓN

| # | Archivo que necesita | Quién lo produce | Dónde está | Cómo se carga | Si no existe |
|---|---|---|---|---|---|
| 1 | `inference.py`, `s03_preprocess.py`, `s04_train_eval.py`, `common.py` | Majo (entrenamiento) | `experimento_dl/` | `import inference` vía `app_core` | No se puede inferir; error explícito |
| 2 | `config/preprocessing.json`, `config/model_{mil,bigru}.json` | Majo | `artifacts/v2/config/` | `app_core.preprocessing_config()` | Estado PENDIENTE |
| 3 | `mil_final.pt`, `bigru_final.pt` (fuera de git) | Majo (Output del notebook de Kaggle) | `artifacts/v2/weights/` | `app_core.load_models()` → `inference.load_classifier` (modo seguro + SHA-256 vs `MANIFEST.json`) | Predicción **PENDIENTE**; no se muestran probabilidades simuladas |
| 4 | `metrics_*.json`, `comparacion_modelos.csv`, `comparacion_pareada.json`, `{val,test}_predictions_*.csv`, `history_*.csv`, `test_slice_probs_mil.npz`, `figures/*.png` | Majo | `artifacts/v2/results/` | `app_core.metrics()`, `predictions()`, … | La sección correspondiente muestra PENDIENTE |
| 5 | Estudio DICOM (todos los cortes axiales de un estudio) | Kaggle RSNA 2022 (cada integrante con su cuenta) | `experimento_dl/data/dicom/<UID>/*.dcm` (no versionado) o subida en la app | `app_core.read_dicoms()` | Se usa `CT_small.dcm` de pydicom (TC real de 1 corte) solo como prueba técnica |

**Entrega a los demás**
- **Méndez (informe):** capturas `docs/app/capturas/*.png` y la sección `docs/app/SECCION_INFORME_APP.md`.
- **Majo (entrenamiento):** incompatibilidades del contrato en `docs/app/APP_STREAMLIT.md` §Incompatibilidades.
- **Hansel (evaluación):** formatos que consume la app en `docs/app/APP_STREAMLIT.md` §Formatos."""),
code("""import sys, os
from pathlib import Path
REPO = Path.cwd().resolve()
while not (REPO / "experimento_dl").exists():
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "experimento_dl"))
import app_core as core   # fija RSNA_EXP=v2 y las rutas de artifacts/v2 ANTES de importar common
import common, inference
print("Configuración:", common.EXP, "· K =", common.K_SLICES, "· contexto ±", common.CONTEXT_MM, "mm")
print("Resultados:", core.RESULTS_DIR.relative_to(REPO), "· pesos:", core.WEIGHTS_DIR.relative_to(REPO))"""),
md("## 1 · Disponibilidad de artefactos y checkpoints"),
code("""core.artifact_status()"""),
code("""import pandas as pd
status = pd.DataFrame([core.checkpoint_status(m) for m in core.MODELS])
status[["nombre", "estado", "existe", "sha256_esperado"]]"""),
md("## 2 · Métricas reales que consume la app (formato del contrato)"),
code("""core.comparison_table()"""),
code("""# Comprobación: las métricas publicadas se reproducen desde las predicciones guardadas
for m in core.MODELS:
    pr, ref = core.predictions(m), core.metrics(m)["test_thr_val_youden"]
    r = core.metrics_at_threshold(pr.patient_overall, pr.prob, ref["threshold"])
    _, auc = core.roc_points(pr.patient_overall, pr.prob)
    ok = (r["tp"], r["fp"], r["tn"], r["fn"]) == (ref["tp"], ref["fp"], ref["tn"], ref["fn"]) and abs(auc - ref["roc_auc"]) < 1e-6
    print(f"{core.NAMES[m]}: AUC={auc:.4f} · matriz {r['tp'], r['fp'], r['tn'], r['fn']} · coincide con metrics_{m}.json: {ok}")"""),
md("""## 3 · Preparar un estudio DICOM

Si hay un estudio real de RSNA en `experimento_dl/data/dicom/<UID>/`, se usa. Si no, se usa `CT_small.dcm`
(corte de TC real incluido en pydicom) **solo como prueba técnica** del flujo; no es un estudio de columna cervical."""),
code("""local = core.local_studies()
if local:
    files = core.local_study_files(local[0]); fuente = f"estudio RSNA local {local[0]}"
else:
    from pydicom.data import get_testdata_file
    files = [get_testdata_file("CT_small.dcm")]; fuente = "CT_small.dcm (prueba técnica, 1 corte)"
dsets, errors, warns = core.read_dicoms(files)
print("Fuente:", fuente, "· cortes legibles:", len(dsets), "· errores:", errors)
for w in warns: print("AVISO:", w)
core.study_summary(core.sort_by_z(dsets))"""),
code("""# Entradas inválidas: deben producir errores explícitos, no predicciones
class F:
    def __init__(s, n, d): s.name, s._d = n, d
    def getvalue(s): return s._d
for caso, items in [("sin archivos", []), ("no DICOM", [F("x.dcm", b"texto")]), ("archivo vacío", [F("v.dcm", b"")])]:
    try:
        core.read_dicoms(items); print(caso, "→ aceptado (inesperado)")
    except core.InvalidInput as e:
        print(f"{caso:14s} → InvalidInput: {e}")"""),
md("## 4 · Compatibilidad del preprocesamiento"),
code("""import numpy as np, matplotlib.pyplot as plt
vol, n_files, pw = core.model_input(dsets)       # inference.preprocess_study con config/preprocessing.json
pre = core.preprocessing_config()
print("Tensor:", vol.shape, vol.dtype, "· avisos:", pw)
checks = {
    "forma [K,3,224,224] del entrenamiento": vol.shape == (pre["k_slices"], 3, pre["img_size"], pre["img_size"]),
    "uint8 (como la caché de entrenamiento)": vol.dtype == np.uint8,
    "K coincide con metrics_*.json": all(core.metrics(m)["input_shape"] == list(vol.shape) for m in core.MODELS),
    "preprocesamiento coincide con metrics_*.json": all(core.metrics(m)["preprocessing"] == pre for m in core.MODELS),
}
# El orden de entrada de los archivos no debe cambiar el tensor
rng = np.random.default_rng(0)
vol_shuf, _, _ = core.model_input([dsets[i] for i in rng.permutation(len(dsets))])
checks["invariante al orden de los archivos"] = bool(np.array_equal(vol, vol_shuf))
for k, v in checks.items(): print(("OK  " if v else "FALLA ") + k)
hu, win = core.display_slice(dsets[0])
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
ax[0].imshow(np.clip(hu, -1024, 2000), cmap="gray"); ax[0].set_title("HU crudo")
for c, t in zip(range(3), ["canal 0 (−z)", "canal 1 (central)", "canal 2 (+z)"]):
    ax[c + 1].imshow(vol[len(vol) // 2, c], cmap="gray"); ax[c + 1].set_title(t)
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()"""),
md("""## 5 · Predicción real (solo si hay checkpoints)

La predicción se ejecuta únicamente con los checkpoints publicados (SHA-256 verificado). Si no están,
se informa **PENDIENTE** y no se genera ningún valor."""),
code("""loaded, skipped = core.load_models(core.MODELS)
for m, why in skipped.items():
    print(f"{core.NAMES[m]}: PENDIENTE ({why}) → copiar {core.checkpoint_path(m).name} en {core.WEIGHTS_DIR.relative_to(REPO)}")
if loaded:
    out = core.predict(dsets, loaded)
    display(pd.DataFrame([{"modelo": core.NAMES[m], "probabilidad": r["probability"], "predicción": r["prediction"],
                           "umbral": r["threshold"], "entrada": r["input_shape"], "segundos": round(r["seconds"], 2),
                           "avisos": "; ".join(r["warnings"])} for m, r in out.items()]))
    print(next(iter(out.values()))["disclaimer"])
    # Comparación A vs B y verificación de reproducibilidad (si el estudio está en val/test)
    uid = str(getattr(dsets[0], "StudyInstanceUID", ""))
    table, summary = core.compare_results(out, uid)
    display(table)
    print({k: v for k, v in summary.items() if k != "evaluacion"}, "· evaluación:", summary["evaluacion"])
else:
    print("\\nInferencia NO ejecutada: no hay checkpoints reales disponibles. No se muestran resultados simulados.")"""),
md("""## 6 · Pruebas funcionales de la app

`experimento_dl/tests/test_app_core.py` (pytest) cubre: configuración v2, carga y consistencia de métricas,
estado PENDIENTE y rechazo de checkpoints con hash distinto, entradas inválidas, estudios mezclados,
forma e invariancia al orden de la entrada, comparación A vs B y verificación de reproducibilidad, flujo
completo de predicción en la interfaz, inferencia por lotes (pesos **aleatorios** en un directorio temporal:
no son resultados) y arranque de la app con `streamlit.testing.AppTest`.
`test_inferencia_real.py` se omite hasta que existan los checkpoints reales y estudios RSNA locales."""),
code("""import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", str(REPO / "experimento_dl" / "tests"), "-q", "--color=no", "-p", "no:cacheprovider"],
                   capture_output=True, text=True, cwd=REPO)
print(r.stdout[-600:])"""),
md("""## Conclusiones

- La app y este notebook usan exactamente `inference.py`/`s03_preprocess` con `config/preprocessing.json`:
  la entrada producida coincide en forma, tipo y parámetros con la usada en entrenamiento.
- Métricas, predicciones guardadas y comparaciones mostradas en la app son las publicadas en `artifacts/v2/`
  (verificadas aquí contra las predicciones).
- **La inferencia sobre estudios nuevos se considera completa solo tras probarla con los checkpoints reales**;
  si la sección 5 indica PENDIENTE, esa prueba sigue pendiente."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, str(Path(__file__).resolve().parents[2] / "notebooks" / "04_integracion_app.ipynb"))
