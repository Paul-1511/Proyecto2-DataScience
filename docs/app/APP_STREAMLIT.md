# Aplicación Streamlit — documentación (P1 · Cabrera)

Rama: `feature/streamlit` · Archivo principal: `experimento_dl/app_streamlit.py` ·
Lógica reutilizable: `experimento_dl/app_core.py` · Notebook: `notebooks/04_integracion_app.ipynb`

## Ejecutar

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 -r experimento_dl/requirements.txt pytest
.venv/bin/streamlit run experimento_dl/app_streamlit.py
```

En Windows usar `.venv\Scripts\...`. En macOS/CPU la instalación de `torch` desde PyPI funciona sin el índice `cu128`.

Para habilitar la predicción, copiar `mil_final.pt` y `bigru_final.pt` (Output del notebook de Kaggle) en
`artifacts/v2/weights/` (ignorado por git). La app verifica el SHA-256 contra `artifacts/v2/MANIFEST.json`.

## Funciones

| Pestaña | Contenido |
|---|---|
| 1 · Datos y EDA | Indicadores de `train.csv`, clases, prevalencia por vértebra (todos / solo positivos), correlación y número de vértebras fracturadas, particiones v2, exploración de metadatos DICOM (`dicom_metadata.csv.gz`) por clase, transfer syntax y control de Rescale |
| 2 · Preprocesamiento | Pasos del pipeline v2, `preprocessing.json` y figuras reales (entrada 2.5D, antes/después, montaje) |
| 3 · Visor DICOM | Subida o estudio local; HU crudo vs ventana ósea, histograma HU, metadatos, comparación de cortes y **entrada real del modelo** `[48,3,224,224]` vía `inference.preprocess_study` |
| 4 · Predicción | Selección **Modelo A, Modelo B o ambos**; medidores con umbral de validación, avisos y descargo; probabilidades por posición (MIL). Sin checkpoints → **PENDIENTE**. Explorador de predicciones guardadas de prueba (n=404) con filtros de acuerdo/desacuerdo |
| 5 · Métricas | Validación / prueba (umbral val) / prueba (0.5): tarjetas, matriz de confusión, ROC, curvas de entrenamiento |
| 6 · Comparación | Barras por conjunto y métrica elegidos, tabla Δ (B − A), DeLong / bootstrap / McNemar, umbral interactivo (exploratorio), ROC con punto de operación, cajas por clase, dispersión A vs B |
| 7 · Estado e integración | Disponibilidad de cada artefacto y checkpoint (SHA-256) |
| 8 · Limitaciones | Alcance no clínico |

Barra lateral: interruptor **gráficas interactivas / figuras estáticas PNG** y selector de **gráficas visibles**
(mostrar u ocultar cada tipo de gráfica en todas las pestañas).

## Formatos que consume la app (para P3 · Hansel)

Si la evaluación independiente produce archivos nuevos, la app los puede mostrar si respetan estos formatos
(en una carpeta nueva, p. ej. `artifacts/v3/`, apuntando `RSNA_ARTIFACTS_DIR`):

| Archivo | Columnas / claves usadas |
|---|---|
| `results/metrics_{mil,bigru}.json` | `val`, `test_thr_val_youden`, `test_thr_0_5` → `threshold, n, tp, fp, tn, fn, sensitivity, specificity, precision, f1, accuracy, roc_auc`; `test_auc_ci95_bootstrap`, `best_epoch`, `epochs_run`, `train_time_s`, `device`, `inference_ms_per_study`, `input_shape`, `preprocessing` |
| `results/{val,test}_predictions_{m}.csv` | `StudyInstanceUID, patient_overall, split, prob, pred` |
| `results/history_{m}.csv` | `epoch, train_loss, val_loss, val_auc` |
| `results/comparacion_pareada.json` | `{val,test}` → `n`, `delong{auc_a, auc_b, diff, p_value}`, `bootstrap_pareado{ci95}`, `mcnemar_umbral_val{p_value_exacto}`, `correlacion_probs_spearman`; `nota` |
| `results/splits_summary.csv` | `split, n, positivos, prop_pos` |
| `results/dicom_metadata.csv.gz` | `StudyInstanceUID, slope, intercept, transfer_syntax, slice_thickness, spacing_z_mm, rows, n_files_study` |
| `results/test_slice_probs_mil.npz` | `uids` [n], `probs` [n, 48] |
| `results/figures/*.png` | `confusion_*`, `roc_*`, `curvas_*`, `comparacion_*`, `entrada_25d_*`, `preproc_antes_despues_*`, `montaje_cortes_*` |

## Incompatibilidades del contrato de inferencia (para Majo)

Ninguna bloquea la app; quedan resueltas o documentadas en `app_core.py`:

1. **`test_slice_probs_mil.npz` guarda `uids` como arreglo de objetos** → `np.load(..., allow_pickle=False)` falla.
   La app lo carga con pickle solo si el SHA-256 coincide con `MANIFEST.json`. Sugerencia para futuras versiones: `uids.astype(str)`.
2. **Conflicto v1/v2 de `common`**: la app anterior importaba `common` con la configuración v1 y `load_classifier`
   habría fallado. `app_core` fija `RSNA_EXP=v2` y las rutas de `artifacts/v2/` antes de cualquier importación.
3. El docstring de `inference.py` usa `data/weights/...`; la ruta canónica es `artifacts/v2/weights/`.
4. `common.py` crea directorios al importarse (`weights/`, `figures/`, `data/dicom/`): inocuo, pero es un efecto secundario.
5. `requirements.txt` y los comandos usan rutas de Windows (`.venv/Scripts`) y `torch` CUDA 12.8; en macOS se usa `torch` de PyPI (CPU).
6. **Pendiente**: entregar `mil_final.pt` y `bigru_final.pt`. Sin ellos la inferencia sobre estudios nuevos no puede validarse.

## Pruebas funcionales (13-oct-2026, macOS, Python 3.12, CPU)

`.venv/bin/python -m pytest experimento_dl/tests -q` → **19 passed**.

| Prueba | Resultado |
|---|---|
| Configuración v2 activa (K=48, ±2 mm) y coherente con `preprocessing.json` | OK |
| Métricas JSON ↔ `comparacion_modelos.csv`; matriz de confusión y AUC recalculadas desde las predicciones | OK (coinciden exactamente) |
| `test_slice_probs_mil.npz` (404 × 48) tras verificar hash | OK |
| Checkpoint ausente → PENDIENTE; checkpoint con hash distinto → rechazado | OK |
| Entradas inválidas: sin archivos, no DICOM, vacío, demasiados archivos, válidos + inválidos, estudios mezclados | OK (errores/avisos explícitos) |
| Entrada del modelo `[48,3,224,224]` uint8, invariante al orden de archivos | OK |
| Mecánica de inferencia MIL y BiGRU con pesos **aleatorios** en directorio temporal (no es un resultado) | OK |
| Arranque de la app (8 pestañas, sin excepciones) y modo estático / gráficas ocultas (`AppTest`) | OK |
| Navegador (Playwright): las 8 pestañas, subida de un DICOM de TC real (`CT_small.dcm`) y cálculo de la entrada | OK tras corregir un error (slider con un único corte) |

**Pendiente:** inferencia con los checkpoints reales y con un estudio completo de RSNA (no hay DICOM de la
competencia localmente; la descarga por API produjo HTTP 429). Para probar: copiar los `.pt` y un estudio en
`experimento_dl/data/dicom/<UID>/` y ejecutar la sección 5 del notebook o la pestaña 4.

## Capturas

`docs/app/capturas/*.png` — regenerar con `experimento_dl/tools/capturas_app.py` (con la app en ejecución).
