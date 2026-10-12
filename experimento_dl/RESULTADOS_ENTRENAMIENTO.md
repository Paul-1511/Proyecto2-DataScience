# Resultados de entrenamiento — RSNA 2022 (ResNet18 + MIL vs ResNet18 + BiGRU)

> **Estado (12-oct-2026): ningún modelo ha sido entrenado todavía.** Este documento recoge la
> configuración experimental fijada *antes* de ver resultados y se completará con los artefactos reales
> del notebook de Kaggle. Toda cifra de desempeño saldrá de `results_v2/*.json|csv`; las secciones sin
> evidencia están marcadas **[PENDIENTE]**.

## 1. Entorno

| | Desarrollo / pruebas técnicas | Entrenamiento previsto |
|---|---|---|
| Hardware | RTX 3060 Laptop 6 GB, Windows 11 | Kaggle Notebook, GPU T4 (16 GB) |
| Software | Python 3.11, torch 2.11.0+cu128, torchvision 0.26.0, pydicom 3.0.2 (ver `requirements.txt`) | versión de Kaggle, registrada en cada `metrics_*.json` (`torch_version`, `device`) |

## 2. Dataset y tamaño de muestra

RSNA 2022 Cervical Spine Fracture Detection, `train.csv` (2019 estudios, 47.6 % con fractura) e
imágenes `train_images/` montadas en Kaggle. Configuración **v2**: todos los estudios.
Estudios efectivamente evaluados por partición: **[PENDIENTE]** (se excluyen solo los que fallen en la
lectura DICOM; ver `results_v2/preprocess_failures.csv`).

## 3. Particiones

`results_v2/splits.csv`: por estudio (= paciente en RSNA 2022), estratificadas por `patient_overall`,
semilla 42, **1292 / 323 / 404** (64/16/20), prevalencia 0.476 en las tres; disjuntas (verificado).
Mismas para ambos modelos. La prueba solo se usa en la evaluación final.
Las particiones v1 (160 estudios, 102/26/32) se conservan sin cambios para la app de demostración.

## 4. Preprocesamiento

DICOM (pydicom; JPEG Lossless con pylibjpeg/gdcm) → HU = pixel·RescaleSlope + RescaleIntercept (por
archivo) → PixelPaddingValue a aire → MONOCHROME1 invertido → ventana ósea L400/W1800 → [0,1] →
224×224 (INTER_AREA) → orden por `ImagePositionPatient[2]`.
**Muestreo v2:** 48 posiciones equiespaciadas en el 5–95 % de los cortes (por número de instancia),
entrada 2.5D (corte central y vecinos a ±2 mm). Justificación y alternativas: `KAGGLE_NOTEBOOK.md`.
Variabilidad observada (12 estudios locales): 3 codificaciones, intercept −1024 y 0, grosor 0.5–1.0 mm,
píxel 0.25–0.53 mm, dirección z vs. instancia no constante. Auditoría completa v2: **[PENDIENTE]**.

## 5. Arquitecturas

- **A — ResNet18 + MIL:** ResNet18 ImageNet por posición → logit por posición → logit de estudio =
  media de los 3 logits más altos. 11.18 M parámetros.
- **B — ResNet18 + BiGRU:** embeddings 512-d por posición (orden z) → GRU bidireccional (128/dirección)
  → [media, máx] → logit de estudio. 11.67 M parámetros.

## 6. Hiperparámetros (idénticos para A y B, fijados a priori)

AdamW (lr 1e-4 backbone, 1e-3 cabeza, wd 1e-4), CosineAnnealing, máx. 15 épocas, early stopping con
paciencia 4 sobre AUC de validación, batch 4 estudios × acumulación 2, AMP fp16,
BCEWithLogits con `pos_weight` = neg/pos de entrenamiento, aumento flip horizontal + desplazamiento
±12 px, semilla 42. Umbral operativo: índice de Youden en validación.

## 7. Métricas reales — [PENDIENTE]

## 8. Gráficas — [PENDIENTE]
(se generarán en `results_v2/figures/`: `confusion_*`, `roc_*`, `curvas_*`, `comparacion_*`, `entrada_25d_*`).

## 9. Comparación de modelos — [PENDIENTE]

## 10. Modelo recomendado — [PENDIENTE]
Criterio fijado de antemano: AUC y equilibrio sensibilidad/especificidad, considerando el solapamiento
de los IC95 % bootstrap; si no hay diferencia clara se recomendará el modelo más simple (A) y se dirá
explícitamente que la diferencia no es concluyente.

## 11. Limitaciones

Clasificador global (no localiza ni identifica vértebra); 48 posiciones no garantizan muestrear la
fractura; sin segmentación de la región cervical; sin remuestreo isotrópico; una sola partición y
semilla; sin validación externa ni clínica.

## 12. Ubicación de checkpoints — [PENDIENTE]
Output de la versión del notebook de Kaggle (`weights/*_final.pt`); no se versionan en GitHub.

## 13. Inferencia

`python inference.py data/weights/mil_final.pt <carpeta_con_dcm_de_un_estudio>` o la API de
`INTEGRACION_STREAMLIT.md`.

## 14. Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| Descarga local por API: HTTP 429 con `Retry-After` creciente, conexiones caídas tras pausas | Migración a Kaggle Notebooks (datos montados); descargas locales detenidas |
| Solo 12 estudios locales (todos de entrenamiento): evaluación imposible | No se entrenó con datos insuficientes; v2 en Kaggle |
| JSON de métricas escrito en cp1252 en Windows (`±` ilegible) | `write_text(..., encoding="utf-8")` en `s04` |
| `s04` cargaba todo en RAM | `DataLoader` con lectura bajo demanda |
| `s05` asumía `download_log.csv` | opcional (no existe en v2) |

## Verificaciones técnicas realizadas (sin valor como resultado)

Con un conjunto sintético de 6 estudios locales (2 por partición, K=8): preprocesamiento 2.5D
(forma `[K,3,224,224]` uint8, centros en orden z, canal central idéntico al 2D), entrenamiento de ambos
modelos sin NaN, early stopping, checkpoints cargables, `s05` y `s07_qc.py` con `all_ok: true`, e
inferencia desde DICOM crudo en orden aleatorio igual a la de evaluación (|Δ| ≤ 1.2e-4).
Ninguna métrica de esas pruebas se reporta.
