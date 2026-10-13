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
| — | *(a completar por el responsable)* | | Integrar `inference.py` y `artifacts/v2/`; la app actual usa la configuración v1 y muestra los modelos como PENDIENTE |

## Informe científico (responsable del informe)

| # | Tarea | Estado | Depende de |
|---|---|---|---|
| — | *(a completar por el responsable)* | | Secciones [PENDIENTE] de `experimento_dl/INFORME_TECNICO.md` con datos de `RESULTADOS_ENTRENAMIENTO.md` |

## Auditoría experimental (responsable de la auditoría)

| # | Tarea | Estado | Depende de |
|---|---|---|---|
| — | *(a completar por el responsable)* | | `artifacts/v2/results/` y comandos de `artifacts/README.md` |
