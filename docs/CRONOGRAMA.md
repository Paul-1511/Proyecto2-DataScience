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
| — | *(a completar por el responsable)* | | Secciones [PENDIENTE] de `experimento_dl/INFORME_TECNICO.md` con datos de `RESULTADOS_ENTRENAMIENTO.md` |

## Auditoría experimental (responsable de la auditoría — Hansel López)

| # | Tarea | Estado | Fecha | Evidencia |
|---|---|---|---|---|
| 1 | Creación de rama `feature/evaluacion` a partir de `main` | ✅ | 13-oct | `git branch feature/evaluacion` |
| 2 | Módulo independiente `metrics_eval.py` (métricas, IC95%, DeLong, McNemar, bootstrap pareado) | ✅ | 13-oct | `experimento_dl/metrics_eval.py` |
| 3 | Pruebas unitarias con datos sintéticos (26 tests, todos passing) | ✅ | 13-oct | `experimento_dl/tests/test_metrics_eval.py` |
| 4 | Verificación independiente de particiones disjuntas (sin fuga de datos) | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §5 |
| 5 | Validación de formato de predicciones (columnas, rangos, UIDs) | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §4 |
| 6 | Recálculo de métricas desde predicciones (tolerancia < 1e-6 vs JSON reportado) | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §7 |
| 7 | Prueba de DeLong: ΔAUC = +0.099, p < 0.001 | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §9.1 |
| 8 | Bootstrap pareado: IC95% [+0.048, +0.147], 100% remuestreos favorecen BiGRU | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §9.2 |
| 9 | McNemar exacto: 66 vs 48 aciertos exclusivos, p = 0.11 | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §9.3 |
| 10 | Análisis de errores (FP, FN, desacuerdos entre modelos) | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §10 |
| 11 | Figuras para el informe (ROC, matrices de confusión, distribuciones) | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §11 |
| 12 | Conclusiones, limitaciones y recomendaciones documentadas | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §13–14 |
| 13 | Sección DEPENDENCIAS E INTEGRACIÓN completada | ✅ | 13-oct | `notebooks/03_evaluacion.ipynb` §15 |
| 14 | Actualización de secciones pendientes en INFORME_TECNICO.md | 🔄 | 13-oct | secciones 7, 8, 11 |
| 15 | Pull request `feature/evaluacion` → `main` | ⏳ | — | pendiente de autorización |
