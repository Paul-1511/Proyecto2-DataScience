# Contrato de integración: modelos entrenados → app Streamlit

Responsable de entrenamiento → responsable de la app. La app **no** debe reimplementar el
preprocesamiento ni la arquitectura: debe usar `inference.py`.

## Artefactos

| Artefacto | Ubicación | Contenido |
|---|---|---|
| `mil_final.pt`, `bigru_final.pt` | Output del notebook de Kaggle → copiar a `experimento_dl/data/weights/` (no versionado) | `state_dict`, `model_name`, `threshold` (Youden en validación), `preprocessing` (K, 2.5D, ventana, tamaño), `hparams`, `best_epoch`, `val_auc` |
| Métricas | `results_v2/metrics_{mil,bigru}.json`, `comparacion_modelos.csv` | validación, prueba (umbral de validación y 0.5), IC95 % AUC, tiempos |
| Predicciones | `results_v2/{val,test}_predictions_{m}.csv` | `StudyInstanceUID, patient_overall, split, prob, pred` |
| Historiales | `results_v2/history_{m}.csv` | pérdida, AUC de validación, tiempo y memoria por época |
| Figuras | `results_v2/figures/` | `confusion_*`, `roc_*`, `curvas_*`, `comparacion_*`, ejemplos de preprocesamiento |
| Probabilidades por posición (MIL) | `results_v2/test_slice_probs_mil.npz` | interpretación, **no** localización |
| Control de calidad | `results_v2/qc_report.json` | debe tener `"all_ok": true` |

## API

```python
from inference import load_classifier, predict_study

clf = load_classifier("data/weights/mil_final.pt")      # una vez (st.cache_resource)
out = predict_study(uploaded_files, clf)                 # lista de st.file_uploader, rutas o bytes
```

`out` (dict):

| clave | tipo | significado |
|---|---|---|
| `probability` | float 0–1 | probabilidad de ≥1 fractura cervical en el estudio |
| `prediction` | 0/1 | `probability >= threshold` |
| `threshold` | float | umbral fijado en validación (no en prueba) |
| `model` | str | `"mil"` o `"bigru"` |
| `n_files`, `n_positions`, `input_shape` | int / list | archivos leídos y tensor usado (p. ej. `[48, 3, 224, 224]`) |
| `slice_probabilities` | list[float] \| None | solo MIL: probabilidad por posición, orden z |
| `warnings` | list[str] | archivos ilegibles, estudios mezclados, menos cortes que K, falta de Rescale |
| `disclaimer` | str | texto a mostrar junto al resultado |

## Requisitos y reglas

- Entrada: **todos** los cortes axiales de **un** estudio (`.dcm`). No requiere etiquetas.
- `load_classifier` fija `RSNA_EXP/K_SLICES/CONTEXT_MM` desde el checkpoint **antes** de importar el
  pipeline. Si la app ya importó `common` con otra configuración (p. ej. la v1 actual), cargar los
  modelos v2 en un proceso aparte o importar `inference` antes que cualquier otro módulo del pipeline;
  si hay conflicto, `load_classifier` lanza un error explícito (no infiere con un preprocesamiento distinto).
- Los dos checkpoints v2 comparten configuración: pueden cargarse juntos para "comparar ambos".
- Mostrar siempre `disclaimer`; nunca presentar la etiqueta real como predicción.
- Verificado (prueba técnica): la inferencia desde DICOM crudo, con archivos en orden arbitrario,
  reproduce la probabilidad de evaluación (|Δ| ≤ 1.2e-4, por fp16).
- Dependencias: `requirements.txt` (torch/torchvision, pydicom + pylibjpeg/gdcm para JPEG Lossless).
- CPU funciona (más lento); GPU opcional.
