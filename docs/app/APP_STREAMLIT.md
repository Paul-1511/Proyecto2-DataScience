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
| 4 · Predicción | Selección **Modelo A, Modelo B o ambos**; modo **un estudio** o **lote de estudios locales**; medidores con umbral de validación, avisos y descargo. **Comparación de la inferencia**: acuerdo A/B y Δ de probabilidad, margen respecto al umbral, percentil del estudio en la distribución de prueba (global y por clase), histograma de prueba con el estudio marcado, probabilidades por posición (MIL), **verificación de reproducibilidad** (si el estudio está en val/test, |Δ| frente a la probabilidad registrada en la evaluación; aviso si está en entrenamiento) y descarga CSV/JSON. Sin checkpoints → **PENDIENTE**. Explorador de predicciones guardadas de prueba (n=404) con filtros de acuerdo/desacuerdo |
| 5 · Métricas | Validación / prueba (umbral val) / prueba (0.5): tarjetas, matriz de confusión, ROC, curvas de entrenamiento |
| 6 · Comparación | Barras por conjunto y métrica elegidos, tabla Δ (B − A), DeLong / bootstrap / McNemar, umbral interactivo (exploratorio) con **IC95 % bootstrap** de sensibilidad y especificidad (`metrics_eval.py`, P3), ROC con punto de operación, cajas por clase, dispersión A vs B |
| 7 · Estado e integración | Disponibilidad de cada artefacto y checkpoint (SHA-256) |
| 8 · Limitaciones | Alcance no clínico |

Barra lateral: interruptor **gráficas interactivas / figuras estáticas PNG** y selector de **gráficas visibles**
(mostrar u ocultar cada tipo de gráfica en todas las pestañas).

Las métricas a un umbral y los intervalos bootstrap se calculan con `experimento_dl/metrics_eval.py`
(módulo independiente de P3), sin duplicar implementación en la app.

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
6. Checkpoints entregados (13-oct) y verificados: tamaño y SHA-256 idénticos a `MANIFEST.json`.
7. Los DICOM de RSNA 2022 no incluyen la etiqueta `Modality`; la app ya no lo trata como "modalidad distinta de CT".

## Pruebas funcionales (13-oct-2026, macOS, Python 3.12, CPU)

`.venv/bin/python -m pytest experimento_dl/tests -q` → **54 passed** (incluye las 26 pruebas de `metrics_eval` de P3
y las de inferencia con checkpoints reales).

| Prueba | Resultado |
|---|---|
| Configuración v2 activa (K=48, ±2 mm) y coherente con `preprocessing.json` | OK |
| Métricas JSON ↔ `comparacion_modelos.csv`; matriz de confusión y AUC recalculadas desde las predicciones (vía `metrics_eval`) | OK (coinciden exactamente) |
| IC95 % bootstrap contienen la estimación puntual publicada | OK |
| `test_slice_probs_mil.npz` (404 × 48) tras verificar hash | OK |
| Checkpoint ausente → PENDIENTE; checkpoint con hash distinto → rechazado | OK |
| Entradas inválidas: sin archivos, no DICOM, vacío, demasiados archivos, válidos + inválidos, estudios mezclados | OK (errores/avisos explícitos) |
| Entrada del modelo `[48,3,224,224]` uint8, invariante al orden de archivos | OK |
| Búsqueda en particiones, percentiles en prueba, detección de no reproducibilidad | OK |
| **Flujo completo en la interfaz** (`AppTest`): estudio local → "Ejecutar predicción" con ambos modelos → panel de comparación, etiqueta rotulada aparte | OK con pesos **aleatorios** (fixture temporal); la verificación marca correctamente "NO reproduce" |
| Inferencia por lotes; un estudio inexistente se informa sin romper el lote | OK |
| Script `tools/validar_inferencia.py` (salida en carpeta temporal) | OK; sin checkpoints termina con PENDIENTE (código 2) |
| Arranque de la app y modo estático / gráficas ocultas; 8 pestañas en navegador sin excepciones | OK |
| **Inferencia con checkpoints reales** (`tests/test_inferencia_real.py`, `tools/validar_inferencia.py`) | **OK** — 2 estudios de prueba RSNA (`…8744` sin fractura, `…9926` con fractura) desde DICOM crudo en CPU: 4/4 probabilidades reproducen la evaluación, |Δ| máx. 6.3e-4 (tol. 1e-3), mismas clases; ~0.9 s por estudio y modelo. Evidencia: `docs/app/validacion_inferencia.{csv,json}`, captura `capturas/04_prediccion_real.png` |

### Cómo repetir la validación real

1. Copiar `mil_final.pt` y `bigru_final.pt` en `artifacts/v2/weights/`.
2. Obtener estudios de **val/test** (UID en `artifacts/v2/results/splits.csv`) en
   `experimento_dl/data/dicom/<StudyInstanceUID>/*.dcm`, p. ej. con
   `experimento_dl/kaggle/descargar_estudio_api.py <UID> …` (API de Kaggle, con pausas y reintentos).
3. Ejecutar:
   ```bash
   .venv/bin/python -m pytest experimento_dl/tests/test_inferencia_real.py -v
   .venv/bin/python experimento_dl/tools/validar_inferencia.py
   ```
   El script deja `docs/app/validacion_inferencia.{csv,json}` (reproducibilidad, tiempos, acuerdo A/B).

## Capturas

`docs/app/capturas/*.png` — regenerar con `experimento_dl/tools/capturas_app.py` (con la app en ejecución).
