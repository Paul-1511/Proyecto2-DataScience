# Artefactos compartidos — experimento v2 (RSNA 2022)

Salida **real y verificada** del entrenamiento en Kaggle (Tesla T4) de dos clasificadores globales por
estudio: **A · ResNet18 + MIL** y **B · ResNet18 + BiGRU**. Fuente técnica completa:
[`docs/RESULTADOS_ENTRENAMIENTO.md`](../docs/RESULTADOS_ENTRENAMIENTO.md).
Modelo recomendado: **B (BiGRU)**, AUC prueba 0.737 [0.688, 0.784] vs 0.638 [0.583, 0.693].
No es una herramienta clínica; no localiza fracturas.

Integridad: `v2/MANIFEST.json` (tamaño y SHA-256 de cada archivo). Verificaciones: `qc_report.json`
(Kaggle), `qc_report_local.json`, `verificacion_metricas.json` → `all_ok: true`.
Archivos de solo lectura: si algo se recalcula, generar una carpeta nueva (p. ej. `v3/`), no editar `v2/`.

## Estructura

```
artifacts/
├── README.md                         (este archivo)
└── v2/
    ├── MANIFEST.json                 tamaño + SHA-256 de todo (incluye los .pt no versionados)
    ├── config/
    │   ├── preprocessing.json        K=48, 2.5D ±2 mm, ventana L400/W1800, 224×224, orden z
    │   ├── model_mil.json            hiperparámetros, umbral, época elegida, AUC val
    │   └── model_bigru.json
    ├── weights/                      NO versionado (*.pt ignorado por git)
    │   ├── mil_final.pt   (44.8 MB)  → obtener del Output del notebook de Kaggle o del responsable
    │   └── bigru_final.pt (46.8 MB)
    └── results/
        ├── splits.csv, splits_summary.csv
        ├── metrics_{mil,bigru}.json
        ├── comparacion_modelos.csv, comparacion_pareada.json
        ├── {val,test}_predictions_{mil,bigru}.csv
        ├── history_{mil,bigru}.csv
        ├── test_slice_probs_mil.npz
        ├── qc_report.json (Kaggle), qc_report_local.json, verificacion_metricas.json
        ├── dicom_metadata.csv.gz, orden_z_auditoria.csv, preprocess_failures.csv, preprocess_times.csv
        ├── INFORME_RESULTADOS.md     (generado automáticamente en Kaggle por s05)
        └── figures/*.png
```

## Formatos

| Archivo | Formato |
|---|---|
| `splits.csv` | `StudyInstanceUID, patient_overall (0/1), split (train/val/test)` — 2019 filas |
| `{val,test}_predictions_<m>.csv` | `StudyInstanceUID, patient_overall, split, prob (0–1), pred (0/1 con el umbral de validación)` |
| `metrics_<m>.json` | claves `val`, `test_thr_val_youden`, `test_thr_0_5` → `{threshold, n, tp, fp, tn, fn, sensitivity, specificity, precision, f1, accuracy, roc_auc}`; `test_auc_ci95_bootstrap` [lo, hi]; `train_time_s`, `inference_ms_per_study`, `best_epoch`, `epochs_run`, `gpu_peak_gb`, `device`, `hparams`, `preprocessing` |
| `comparacion_modelos.csv` | una fila por modelo × conjunto (validación / prueba umbral val / prueba umbral 0.5) |
| `comparacion_pareada.json` | por conjunto: `delong` {auc_a (MIL), auc_b (BiGRU), diff, z, p_value}, `bootstrap_pareado` {ci95}, `mcnemar_umbral_val` |
| `history_<m>.csv` | `epoch, train_loss, val_loss, val_auc, epoch_s, elapsed_s, lr_backbone, gpu_peak_gb` |
| `test_slice_probs_mil.npz` | `uids` [404], `probs` [404, 48] (orden z; interpretación, no localización) |
| `dicom_metadata.csv.gz` | un registro por corte central: rescale, fotometría, transfer syntax, spacing, grosor, z, paso 2.5D |
| `*_final.pt` | `torch.save` dict: `state_dict, model_name, threshold, preprocessing, hparams, best_epoch, val_auc, torch_version, note`. Cargar con `inference.load_classifier` (modo seguro) |

## Qué necesita cada rol

> Los roles los definió el equipo (app Streamlit, informe científico, auditoría experimental);
> este documento no asume qué integrante tiene cada uno.

**App Streamlit**
- `v2/weights/{mil,bigru}_final.pt` (fuera de git) + `experimento_dl/inference.py`
  (`load_classifier`, `predict_study`) — contrato en `docs/INTEGRACION_STREAMLIT.md`.
- Para mostrar resultados: `metrics_*.json`, `comparacion_modelos.csv`, `comparacion_pareada.json`,
  `history_*.csv`, `test_predictions_*.csv`, `figures/` (todas reales).
- Importante: los modelos v2 usan 48 posiciones 2.5D; la app actual (v1, 24 cortes 2D) no debe
  reutilizar su preprocesamiento para ellos — `inference.py` lo resuelve desde el checkpoint.

**Informe científico**
- `docs/RESULTADOS_ENTRENAMIENTO.md` (tablas listas), `comparacion_modelos.csv`,
  `comparacion_pareada.json`, figuras `comparacion_*.png`, `confusion_*.png`, `roc_*.png`,
  `curvas_*.png`, `entrada_25d_*.png`, `preproc_antes_despues_*.png`, `config/*.json`.
- Secciones [PENDIENTE] de `experimento_dl/INFORME_TECNICO.md` (resultados, comparación, selección, conclusiones).

**Auditoría experimental**
- `splits.csv` (+ `experimento_dl/s01_sample_split.py` para regenerarlo: semilla 42).
- Predicciones `{val,test}_predictions_*.csv` y `metrics_*.json` para recalcular métricas;
  `experimento_dl/s08_verificacion.py` y `s07_qc.py` (ejecutar con
  `RSNA_EXP=v2 RSNA_RESULTS_DIR=artifacts/v2/results RSNA_WEIGHTS_DIR=artifacts/v2/weights`).
- `qc_report.json` (Kaggle; incluye inferencia desde DICOM crudo), `verificacion_metricas.json`,
  `MANIFEST.json` (hashes), `preprocess_failures.csv`, `orden_z_auditoria.csv`, `dicom_metadata.csv.gz`.

## Reproducir las verificaciones

```bash
cd experimento_dl
RSNA_EXP=v2 RSNA_RESULTS_DIR=../artifacts/v2/results RSNA_WEIGHTS_DIR=../artifacts/v2/weights \
  ../.venv/Scripts/python s07_qc.py --n-infer 0 --out qc_report_local.json
RSNA_EXP=v2 RSNA_RESULTS_DIR=../artifacts/v2/results ../.venv/Scripts/python s08_verificacion.py
```
