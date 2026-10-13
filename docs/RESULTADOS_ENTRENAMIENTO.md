# Resultados de entrenamiento — RSNA 2022 (ResNet18 + MIL vs ResNet18 + BiGRU)

> **Estado (13-oct-2026): ambos modelos entrenados y evaluados en Kaggle (GPU Tesla T4).**
> Todas las cifras provienen de `artifacts/v2/results/` (copia verificada de la salida de Kaggle).
> Verificaciones: `qc_report.json` (Kaggle, incluye inferencia desde DICOM crudo),
> `qc_report_local.json` y `verificacion_metricas.json` (local) → `all_ok: true`.
> Es un clasificador **global por estudio**: no detecta ni localiza fracturas y **no tiene validación clínica**.

## 1. Entorno

| | Entrenamiento y evaluación | Verificación local |
|---|---|---|
| Hardware | Kaggle Notebook, **Tesla T4** (16 GB), 4 CPU | RTX 3060 Laptop 6 GB |
| Software | Python 3, torch 2.11.0+cu128 (registrado en cada `metrics_*.json`) | torch 2.11.0+cu128, pydicom 3.0.2 |
| Código | rama `feature/entrenamiento` @ `772a276`, notebook `kaggle/rsna_entrenamiento_v2.ipynb` | `s07_qc.py`, `s08_verificacion.py` |

## 2. Dataset y tamaño de muestra

RSNA 2022 Cervical Spine Fracture Detection: **todos** los estudios de `train.csv` (2019; 47.6 % con
fractura), DICOM leídos desde `/kaggle/input`. Preprocesamiento: **2019/2019 estudios sin errores**
(`preprocess_failures.csv` vacío), metadatos auditados de 96 912 cortes centrales (2019 × 48; los vecinos 2.5D se leen pero no se registran),
~2.5 s/estudio (≈21 min con 4 procesos).

## 3. Particiones

`artifacts/v2/results/splits.csv` (idéntico a `experimento_dl/results_v2/splits.csv`): por estudio
(= paciente), estratificado por `patient_overall`, semilla 42.

| partición | estudios | con fractura | evaluados |
|---|---|---|---|
| entrenamiento | 1292 | 615 | 1292 |
| validación | 323 | 154 | 323 |
| prueba | 404 | 192 | 404 |

Disjuntas (verificado). Mismas para ambos modelos. La prueba no se usó para elegir épocas, umbrales ni
configuración.

## 4. Preprocesamiento

HU = pixel·RescaleSlope + RescaleIntercept (por archivo) → PixelPaddingValue a aire → MONOCHROME1
invertido → ventana ósea L400/W1800 → [0,1] → 224×224 → orden por `ImagePositionPatient[2]`.
48 posiciones equiespaciadas (5–95 % de los cortes por número de instancia), entrada 2.5D (vecinos a
±2 mm, desplazamiento = round(2 mm / espaciado z)). Configuración: `artifacts/v2/config/preprocessing.json`.

Auditoría sobre los 2019 estudios (`dicom_metadata.csv.gz`, `orden_z_auditoria.csv`):

| Aspecto | Observado |
|---|---|
| Codificación (cortes centrales) | Explicit VR LE 59 616 cortes, JPEG Lossless 22 128, Implicit VR LE 15 072, **JPEG 2000 96** (2 estudios); todas decodificadas |
| Rescale | slope 1 / intercept −1024 (mayoría), slope 1 / intercept 0, **slope 10 / intercept −10240** (141 estudios), intercept −8192 (3 estudios) |
| Fotometría | 100 % MONOCHROME2; PixelPaddingValue ausente en todos |
| Grosor de corte | 0.49–1.0 mm (mediana 0.625) |
| Espaciado z | 0.31–2.63 mm (mediana 0.625) → desplazamiento 2.5D de 1 a 6 cortes (mediana 3) |
| Cortes por estudio | 69–1082 (mediana 314) |
| Orden z vs instancia | 1821 decreciente, 197 creciente, **1 mixto** (no monótono); z únicos en 2019/2019 |

Las diferencias de rescale, grosor y espaciado son parámetros de **adquisición/reconstrucción**, no
diferencias clínicas; el pipeline las normaliza por archivo. Pendiente: inspección visual de un estudio
con slope 10 y de los 3 con intercept −8192 (no disponibles localmente) para confirmar rangos HU.

## 5. Arquitecturas

- **A — ResNet18 + MIL** (11.18 M parámetros): ResNet18 ImageNet por posición → logit por posición →
  logit de estudio = media de los 3 logits más altos.
- **B — ResNet18 + BiGRU** (11.67 M parámetros): embeddings 512-d por posición en orden z →
  GRU bidireccional (128/dirección) → [media, máx] → logit de estudio.

## 6. Hiperparámetros (idénticos, fijados antes de entrenar)

AdamW (lr 1e-4 backbone, 1e-3 cabeza, wd 1e-4), CosineAnnealing (T_max = 15), máx. 15 épocas,
early stopping (paciencia 4, AUC de validación), batch 4 estudios × acumulación 2, AMP fp16,
BCEWithLogits con `pos_weight` = 1.1008 (neg/pos de entrenamiento), aumento flip horizontal +
desplazamiento ±12 px, semilla 42. Umbral: índice de Youden en validación.
Detalle por modelo: `artifacts/v2/config/model_{mil,bigru}.json`.

## 7. Métricas reales

**Prueba (n = 404; 192 con fractura), umbral fijado en validación:**

| Modelo | ROC-AUC [IC95 % bootstrap] | Sensibilidad | Especificidad | Precisión | F1 | Exactitud | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|
| A · MIL (umbral 0.550) | **0.638** [0.583, 0.693] | 0.365 | 0.816 | 0.642 | 0.465 | 0.601 | 173/39/122/70 |
| B · BiGRU (umbral 0.548) | **0.737** [0.688, 0.784] | 0.641 | 0.651 | 0.624 | 0.632 | 0.646 | 138/74/69/123 |

**Prueba con umbral 0.5 (referencia, no elegido):** MIL sens 0.516 / esp 0.708 / F1 0.561;
BiGRU sens 0.714 / esp 0.590 / F1 0.659.

**Validación (n = 323; 154 con fractura) — usada para elegir época y umbral, por tanto optimista:**

| Modelo | ROC-AUC | Sensibilidad | Especificidad | F1 | Mejor época / ejecutadas |
|---|---|---|---|---|---|
| A · MIL | 0.687 | 0.481 | 0.811 | 0.569 | 6 / 10 |
| B · BiGRU | 0.725 | 0.721 | 0.657 | 0.687 | 5 / 9 |

**Tiempos y recursos (Tesla T4):**

| Modelo | Entrenamiento | s/época | Inferencia (prueba, GPU) | VRAM pico |
|---|---|---|---|---|
| A · MIL | 13.7 min | ~81 s | 28.0 ms/estudio | 3.21 GB |
| B · BiGRU | 12.4 min | ~81 s | 23.8 ms/estudio | 3.21 GB |

La inferencia excluye la lectura y el preprocesamiento DICOM (~2.5 s/estudio en CPU), que dominan el
tiempo de extremo a extremo.

## 8. Gráficas (`artifacts/v2/results/figures/`)

`comparacion_roc_prueba.png`, `comparacion_metricas_prueba.png` (incluye sensibilidad/especificidad),
`comparacion_tiempos.png`, `comparacion_probabilidades_prueba.png`, `comparacion_auc_validacion.png`,
`confusion_{mil,bigru}.png`, `roc_{mil,bigru}.png`, `curvas_{mil,bigru}.png`,
`entrada_25d_*.png`, `preproc_antes_despues_*.png`, `montaje_cortes_*.png`.

## 9. Comparación de modelos (`comparacion_pareada.json`)

Mismo conjunto de prueba; diferencias = BiGRU − MIL.

| Conjunto | ΔAUC | DeLong | Bootstrap pareado (IC95 %) | McNemar (umbral val) |
|---|---|---|---|---|
| Prueba (404) | **+0.099** | z = 4.09, p = 4.3·10⁻⁵ | [+0.048, +0.147]; 0 % de remuestreos ≤ 0 | 66 vs 48 aciertos exclusivos, p = 0.11 |
| Validación (323) | +0.038 | z = 1.58, p = 0.115 | [−0.010, +0.086] | 50 vs 39, p = 0.29 |

Lectura: en prueba, BiGRU discrimina mejor que MIL y la diferencia de AUC es estadísticamente
significativa en la comparación pareada (los IC95 % individuales apenas se solapan: 0.688 vs 0.693).
En validación la dirección es la misma, pero sin significación. Con los umbrales de validación, la
diferencia en exactitud no es significativa (McNemar); MIL opera con baja sensibilidad (0.365) y
BiGRU con un punto más equilibrado (0.641 / 0.651).
Las probabilidades de ambos modelos están moderadamente correlacionadas (Spearman 0.60 en prueba).

## 10. Modelo recomendado: **B · ResNet18 + BiGRU**

Criterio fijado antes de entrenar: AUC y equilibrio sensibilidad/especificidad, considerando la
incertidumbre. BiGRU es superior en AUC de validación (0.725 vs 0.687) y de prueba (0.737 vs 0.638;
diferencia pareada +0.099, IC95 % [0.048, 0.147]) y tiene un punto operativo más equilibrado; además
su AUC fue estable entre validación y prueba, mientras que el de MIL cayó (0.687 → 0.638) y fluctuó
mucho entre épocas (0.56–0.69). Matices: la ventaja en validación no fue significativa por sí sola,
es una única partición y semilla, y el desempeño absoluto es **moderado** (AUC 0.74; ~36 % de falsos
negativos y ~35 % de falsos positivos en prueba), insuficiente para cualquier uso clínico.

## 11. Limitaciones

- Desempeño moderado; el umbral de Youden equilibra errores pero no prioriza la sensibilidad, que
  sería lo deseable en un cribado (no se ajustó, por diseño, para no usar la prueba).
- Sobreajuste rápido: la pérdida de entrenamiento baja continuamente y la de validación sube desde la
  época 5–6; el early stopping lo contiene, pero el AUC de validación varía ±0.03 entre épocas
  (selección de época con ruido; la validación queda optimista).
- 48 posiciones por estudio no garantizan muestrear la fractura; sin segmentación de C1–C7, parte de
  las posiciones cae en cráneo/tórax.
- Etiqueta global por estudio: no se usan las etiquetas por vértebra ni las cajas.
- Sin remuestreo isotrópico; espaciado z variable (0.31–2.63 mm).
- Una partición y una semilla; sin validación cruzada ni externa; sin validación clínica.
- Las probabilidades por posición del modelo MIL (`test_slice_probs_mil.npz`) son interpretativas,
  no una localización validada.
- Métricas no comparables con el leaderboard de la competencia (métrica ponderada por vértebra, test oculto).

## 12. Ubicación de checkpoints

| Archivo | Tamaño | SHA-256 (inicio) | Dónde |
|---|---|---|---|
| `mil_final.pt` | 44.8 MB | `d49e18c4effd476b` | Output de la versión del notebook en Kaggle; copia local en `artifacts/v2/weights/` (no versionada) |
| `bigru_final.pt` | 46.8 MB | `14b4c2ee511854ee` | ídem |

Los hashes completos están en `artifacts/v2/MANIFEST.json`. Nota: al descargarlos de Kaggle llegaron con
extensión `.zip` (es el formato interno de `torch.save`); basta renombrarlos a `.pt`.

## 13. Inferencia

```bash
python experimento_dl/inference.py artifacts/v2/weights/bigru_final.pt <carpeta_con_los_dcm_de_un_estudio>
```

API y formato de salida: `docs/INTEGRACION_STREAMLIT.md`. Carga segura
(`weights_only=True`). Verificado en Kaggle: la inferencia desde DICOM crudo reproduce las
probabilidades de evaluación en 5 estudios de prueba por modelo (|Δ| ≤ 9·10⁻⁵).

## 14. Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| Descarga local por API: HTTP 429 con `Retry-After` creciente | Migración a Kaggle Notebooks (datos montados) |
| Solo 12 estudios locales: evaluación imposible | No se entrenó localmente; v2 completo en Kaggle |
| JSON de métricas en cp1252 en Windows | `encoding="utf-8"` en `s04` |
| `s04` cargaba todo en RAM | `DataLoader` con lectura bajo demanda |
| Checkpoints con `TorchVersion` no cargaban con `weights_only=True` | allowlist explícita de esa clase en `inference.py`; `s04` guarda ahora la versión como texto |
| Al recalcular métricas desde CSV, 1 estudio de validación por modelo cambiaba de clase | Su probabilidad es exactamente el umbral de Youden y el CSV (float32 a 8 dígitos) la deja ~2·10⁻⁸ por debajo; las métricas del JSON son correctas. `s08` compara con tolerancia 1e-6; `s04` escribe ahora probabilidades en float64. Prueba no afectada (0 estudios en el límite) |
| Descarga de Kaggle con extensión `.zip` en los checkpoints | Renombrar a `.pt`; integridad verificada por carga estricta y SHA-256 |
