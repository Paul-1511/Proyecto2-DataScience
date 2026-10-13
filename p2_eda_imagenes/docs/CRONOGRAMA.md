# Cronograma del proyecto final — RSNA 2022 (CC3084 Data Science)

Archivo compartido. Cada responsable actualiza **solo su sección**. Estados: ✅ hecho · 🔄 en curso · ⏳ pendiente · ⛔ bloqueado.

## Entrenamiento, evaluación y GPU (responsable del componente computacional)

| # | Tarea | Estado | Fecha | Evidencia |
|---|---|---|---|---|
| 1 | Diagnóstico del entorno (GPU RTX 3060, PyTorch CUDA en `.venv`) | ✅ | 12-oct | `experimento_dl/requirements.txt` |
| 2 | Pipeline DICOM: HU, ventana ósea, 224×224, orden por `ImagePositionPatient` | ✅ | 12-oct | `experimento_dl/s03_preprocess.py` |
| 3 | Descarga local por API de Kaggle | ⛔→ descartada | 12-oct | HTTP 429 con `Retry-After` creciente; ver `RESULTADOS_ENTRENAMIENTO.md` §14 |
| 4 | Migración a Kaggle Notebooks (notebook, guía, paquete de código) | ✅ | 12-oct | `experimento_dl/kaggle/` |
| 5 | Revisión del muestreo (v2: 48 posiciones 2.5D, 2019 estudios) | ✅ | 12-oct | `experimento_dl/KAGGLE_NOTEBOOK.md` |
| 6 | Particiones v2 1292/323/404 (64/16/20), estratificadas, semilla 42 | ✅ | 12-oct | `artifacts/v2/results/splits.csv` |
| 7 | Entrenamiento Modelo A · ResNet18 + MIL (Kaggle T4) | ✅ | 13-oct | `artifacts/v2/results/metrics_mil.json` |
| 8 | Entrenamiento Modelo B · ResNet18 + BiGRU (Kaggle T4) | ✅ | 13-oct | `artifacts/v2/results/metrics_bigru.json` |
| 9 | Evaluación en prueba (n=404), comparación pareada, selección (BiGRU) | ✅ | 13-oct | `comparacion_pareada.json`, `RESULTADOS_ENTRENAMIENTO.md` |
| 10 | Control de calidad (`s07_qc.py`, `s08_verificacion.py`, MANIFEST SHA-256) | ✅ | 13-oct | `qc_report*.json`, `verificacion_metricas.json` → `all_ok` |
| 11 | Inferencia reutilizable y contrato con la app | ✅ | 13-oct | `experimento_dl/inference.py`, `INTEGRACION_STREAMLIT.md` |
| 12 | Artefactos compartidos y guía por rol | ✅ | 13-oct | `artifacts/README.md` |
| 13 | Pull request `feature/entrenamiento` → `main` | 🔄 | 13-oct | pendiente de revisión y autorización de merge |
| 14 | Entrega de checkpoints (`*.pt`, fuera de git) al responsable de la app | ⏳ | — | Output del notebook de Kaggle; hashes en `artifacts/v2/MANIFEST.json` |
| 15 | Inspección visual de estudios con RescaleSlope 10 / intercept −8192 | ⏳ | — | requiere sesión de Kaggle (no hay píxeles locales) |

## App Streamlit (responsable de la app)

| # | Tarea | Estado | Depende de |
|---|---|---|---|
| 1 | Migración de la app a v2 (`artifacts/v2/`, `inference.py`, sin conflicto v1/v2) | ✅ 13-oct | `experimento_dl/app_core.py`, `app_streamlit.py` |
| 2 | Selección Modelo A / B / ambos; estado PENDIENTE sin checkpoints; SHA-256 vs MANIFEST | ✅ 13-oct | pestaña 4 |
| 3 | EDA y metadatos DICOM interactivos; visor DICOM conservado + entrada real 2.5D | ✅ 13-oct | pestañas 1–3 |
| 4 | Comparación interactiva de métricas, pruebas pareadas, umbral exploratorio; controles para mostrar/ocultar gráficas | ✅ 13-oct | pestañas 5–6 |
| 5 | Validación de entradas inválidas y errores de carga | ✅ 13-oct | `experimento_dl/tests/test_app_core.py` (19 passed) |
| 6 | Notebook de integración con salidas reales | ✅ 13-oct | `notebooks/04_integracion_app.ipynb` |
| 7 | Capturas, documentación y sección del informe → Méndez | ✅ 13-oct | `docs/app/` |
| 8 | Incompatibilidades del contrato → Majo; formatos de métricas → Hansel | ✅ 13-oct | `docs/app/APP_STREAMLIT.md` |
| 9 | Inferencia con checkpoints reales y estudio RSNA completo | ⏳ | Requiere `mil_final.pt`, `bigru_final.pt` (Majo) y un estudio DICOM |
| 10 | Pull request `feature/streamlit` → `main` | ⏳ | pendiente de autorización |

## Informe científico (responsable del informe)

| # | Tarea | Estado | Depende de |
|---|---|---|---|
| 1 | Script de metadatos DICOM independiente (`scripts/eda_metadatos_dicom.py`), probado con la auditoría real `artifacts/v2/results/dicom_metadata.csv.gz` | ✅ 13-oct | — |
| 2 | Notebook `notebooks/02_eda_imagenes.ipynb` (generado por `notebooks/tools/build_notebook_02.py`) | 🔄 código listo; **sin ejecutar en Kaggle** | Sesión de Kaggle con el dataset de la competencia |
| 3 | Informe de calidad DICOM con cifras reales de la auditoría (2019 estudios, 48 cortes centrales) | ✅ 13-oct | `artifacts/eda_imagenes/auditoria_v2/` |
| 4 | Informe de calidad DICOM con cabeceras de todos los cortes + intensidades HU + antes/después | ⏳ | Ejecución del notebook 02 en Kaggle |
| 5 | Informe LaTeX (integración editorial) | ⏳ | No incluido todavía en esta rama |
| 6 | Integrar §7–9 y conclusiones con resultados y capturas | ⏳ | Majo (metodología), Cabrera (capturas), Hansel (comparación/conclusiones) |
| 7 | Verificación de referencias APA y rúbrica | ⏳ | Cuando exista el LaTeX |
| 8 | Pull request `feature/eda-imagenes` → `main` | ⏳ | pendiente de push y autorización |

## Auditoría experimental (responsable de la auditoría)

| # | Tarea | Estado | Depende de |
|---|---|---|---|
| — | *(a completar por el responsable)* | | `artifacts/v2/results/` y comandos de `artifacts/README.md` |
