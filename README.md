# RSNA 2022 — Detección de Fracturas Cervicales en TC

**Proyecto Final — CC3084 Data Science, Universidad del Valle de Guatemala**
**Semestre II — Octubre 2026**

---

## Equipo

| Integrante | Rol | Carnet |
|------------|-----|--------|
| María José Yee | P1: Entrenamiento, checkpoints, predicciones | 231193 |
| Pablo Andrés Cabrera | P2: Aplicación Streamlit e inferencia | 231156 |
| Pablo José Méndez | P3: EDA visual DICOM, informe | 23975 |
| Hansel André López | P4: Evaluación independiente, métricas, comparación | 19026 |

---

## Problema

Clasificar estudios de tomografía computarizada (TC) cervical para determinar si un paciente presenta al menos una fractura cervical (`patient_overall`). Se usan 2,019 estudios del dataset [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection).

> **Nota:** Es un clasificador global por estudio. No detecta ni localiza fracturas, y no tiene validación clínica.

---

## Modelos

| Modelo | Parámetros | Descripción |
|--------|------------|-------------|
| **A — ResNet18 + MIL** | 11.18M | ResNet18 ImageNet por corte → media de top-3 logits |
| **B — ResNet18 + BiGRU** | 11.67M | ResNet18 → embeddings 512-d → GRU bidireccional → [media, max] |

**Modelo seleccionado:** BiGRU (AUC = 0.737, DeLong p < 0.001 vs MIL)

### Resultados en prueba (n = 404)

| Métrica | MIL (top-3) | BiGRU |
|---------|-------------|-------|
| ROC-AUC | 0.638 [0.583, 0.693] | **0.737** [0.688, 0.784] |
| Sensibilidad | 0.365 | **0.641** |
| Especificidad | **0.816** | 0.651 |
| F1 | 0.465 | **0.632** |
| Exactitud | 0.601 | **0.646** |

---

## Estructura del Repositorio

```
Proyecto2-DataScience/
├── README.md                          ← este archivo
├── train.csv                          ← etiquetas (2,019 estudios)
├── train_bounding_boxes.csv           ← anotaciones de cajas
├── docs/
│   ├── CRONOGRAMA.md                  ← estado de tareas por integrante
│   ├── ANALISIS_CUMPLIMIENTO.md       ← verificación vs rúbrica
│   ├── INFORME_TECNICO.md (→ experimento_dl/)
│   ├── Proyecto 2. Resultados. 2026.pdf
│   ├── Proyecto 2. Análisis Exploratorio. 2026.pdf
│   └── app/                           ← documentación de Streamlit
├── experimento_dl/
│   ├── README.md                      ← documentación del módulo
│   ├── common.py                      ← configuración compartida
│   ├── inference.py                   ← API de inferencia
│   ├── metrics_eval.py                ← módulo de evaluación (P4)
│   ├── app_streamlit.py               ← aplicación Streamlit
│   ├── app_core.py                    ← lógica de la app
│   ├── requirements.txt
│   ├── s01_sample_split.py            ← particiones
│   ├── s02_download.py                ← descarga DICOM
│   ├── s03_preprocess.py              ← preprocesamiento
│   ├── s04_train_eval.py              ← entrenamiento
│   ├── s05_compare_report.py          ← comparación y figuras
│   ├── s07_qc.py                      ← control de calidad
│   ├── s08_verificacion.py            ← verificación independiente
│   ├── tests/                         ← pruebas unitarias
│   ├── kaggle/                        ← notebook y paquete para Kaggle
│   └── tools/                         ← scripts auxiliares
├── notebooks/
│   ├── 03_evaluacion.ipynb            ← evaluación independiente (P4)
│   └── 04_integracion_app.ipynb       ← integración con Streamlit (P2)
├── artifacts/
│   ├── README.md                      ← guía de artefactos
│   └── v2/
│       ├── MANIFEST.json              ← integridad (SHA-256)
│       ├── config/                    ← hiperparámetros y preprocesamiento
│       ├── results/                   ← métricas, predicciones, figuras
│       └── weights/                   ← checkpoints (no versionados)
└── imgs/                              ← figuras del EDA inicial
```

---

## Reproducción

### Entorno

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r experimento_dl/requirements.txt
```

### Pipeline completo

```bash
cd experimento_dl
export RSNA_EXP=v2

# 1. Particiones
python3 s01_sample_split.py

# 2. Descarga (requiere KAGGLE_API_TOKEN)
python3 s02_download.py

# 3. Preprocesamiento DICOM
python3 s03_preprocess.py

# 4. Entrenamiento
python3 s04_train_eval.py --model mil
python3 s04_train_eval.py --model bigru

# 5. Comparación y reporte
python3 s05_compare_report.py

# 6. Control de calidad
python3 s07_qc.py
python3 s08_verificacion.py
```

### Aplicación Streamlit

```bash
streamlit run experimento_dl/app_streamlit.py
```

### Pruebas unitarias

```bash
cd experimento_dl
python3 -m pytest tests/ -v
```

---

## Artefactos

Los resultados del entrenamiento están en `artifacts/v2/results/`:

| Archivo | Descripción |
|---------|-------------|
| `metrics_{mil,bigru}.json` | Métricas completas con IC95% |
| `comparacion_modelos.csv` | Tabla comparativa |
| `comparacion_pareada.json` | Pruebas estadísticas (DeLong, McNemar, bootstrap) |
| `{val,test}_predictions_{mil,bigru}.csv` | Predicciones por estudio |
| `history_{mil,bigru}.csv` | Curvas de entrenamiento |
| `verificacion_metricas.json` | Verificación independiente |
| `qc_report.json` | Control de calidad |
| `figures/*.png` | 27 visualizaciones |

Integridad: `artifacts/v2/MANIFEST.json` (SHA-256).

---

## Documentación

- [Informe técnico](experimento_dl/INFORME_TECNICO.md)
- [Resultados de entrenamiento](docs/RESULTADOS_ENTRENAMIENTO.md)
- [Guía de artefactos](artifacts/README.md)
- [Cronograma](docs/CRONOGRAMA.md)
- [Integración Streamlit](docs/INTEGRACION_STREAMLIT.md)

---

## Ramas

| Rama | Descripción |
|------|-------------|
| `main` | Versión estable integrada |
| `feature/evaluacion` | Evaluación independiente (P4) |
| `feature/entrenamiento` | Pipeline v2 y entrenamiento en Kaggle |
| `feature/streamlit` | Aplicación Streamlit v2 |

---

## Licencia y Restricciones

Este proyecto es académico. Los modelos no tienen validación clínica y no deben usarse para diagnóstico médico.

Dataset: [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection) — sujeto a las reglas de la competencia de Kaggle.
