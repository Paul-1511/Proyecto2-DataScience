---
title: "Clasificación global de fracturas cervicales en TC a partir de una muestra reducida de RSNA 2022"
subtitle: "Proyecto final — CC3084 Data Science, Universidad del Valle de Guatemala"
author: "Pablo Andrés Cabrera (231156) · Hansel André López (19026) · Pablo José Méndez (23975) · María José Yee (231193)"
date: "octubre de 2026"
lang: es
---

> **Estado del documento.** Las secciones marcadas **[PENDIENTE]** dependen de ejecuciones que aún no
> habían terminado al redactar esta versión. Todos los números de resultados se toman de
> `experimento_dl/results/` (JSON/CSV generados por los scripts); ningún valor proviene de la literatura.

# 1. Introducción

Las fracturas de la columna cervical son lesiones potencialmente graves: un diagnóstico tardío puede
asociarse a deterioro neurológico. En pacientes adultos con trauma, la tomografía computarizada (TC)
es la modalidad de imagen de referencia para su evaluación, y la decisión de solicitarla suele
apoyarse en reglas clínicas como NEXUS (Hoffman et al., 2000) y la Canadian C-Spine Rule
(Stiell et al., 2001). La lectura de cientos de cortes por estudio es demandante y motivó la
competencia **RSNA 2022 Cervical Spine Fracture Detection** (Flanders et al., 2022), cuyo conjunto de
datos se describe en Lin et al. (2023).

Este informe continúa el análisis exploratorio del Proyecto 2 (datos tabulares `train.csv` y
`train_bounding_boxes.csv`) y lo extiende con un **experimento de aprendizaje profundo reproducible
sobre las imágenes DICOM**, deliberadamente pequeño por restricciones de cómputo, red y tiempo.

# 2. Problema científico

¿Es posible construir, con recursos de cómputo modestos (GPU de 6 GB) y una muestra reducida y
reproducible de estudios de TC de RSNA 2022, un clasificador que estime a **nivel de estudio** la
presencia de al menos una fractura cervical (`patient_overall`), y aporta el modelado secuencial de
los cortes (BiGRU) una mejora medible frente a una agregación MIL simple?

# 3. Objetivos

**General.** Implementar y evaluar sobre estudios no vistos un pipeline reproducible de
preprocesamiento DICOM y clasificación global de fracturas cervicales en TC.

**Específicos.**

1. Construir una muestra estratificada de 160 estudios con particiones por estudio sin fuga
   (entrenamiento 102, validación 26, prueba 32).
2. Implementar el preprocesamiento DICOM (HU, ventana ósea, normalización, 224×224, orden por
   `ImagePositionPatient`) y auditar la variabilidad de metadatos.
3. Entrenar y evaluar ResNet18 + MIL y ResNet18 + BiGRU con las mismas particiones, reportando
   sensibilidad, especificidad, precisión, F1, exactitud y ROC-AUC con IC95 %.
4. Seleccionar un modelo con base en resultados de validación/prueba, explicitando la incertidumbre.
5. Desarrollar una aplicación Streamlit que use el mismo preprocesamiento para inferencia.

# 4. Marco teórico

**TC y unidades Hounsfield.** Los valores almacenados en DICOM se convierten a unidades Hounsfield
(HU) con `HU = valor × RescaleSlope + RescaleIntercept`; el aire ≈ −1000 HU, el agua 0 HU y el hueso
cortical supera varios cientos de HU. El *windowing* recorta un intervalo [L − W/2, L + W/2] para
maximizar el contraste del tejido de interés; aquí se usa una ventana ósea L = 400, W = 1800.

**Redes residuales.** ResNet (He et al., 2016) introduce conexiones residuales que facilitan entrenar
redes profundas; ResNet18 preentrenada en ImageNet (Deng et al., 2009) es un extractor de
características ligero adecuado para 6 GB de VRAM.

**Aprendizaje de múltiples instancias (MIL).** Cuando solo hay etiqueta a nivel de "bolsa"
(estudio) y no de "instancia" (corte), MIL agrega puntuaciones de instancias en una puntuación de
bolsa (Ilse et al., 2018). Un estudio positivo puede tener la mayoría de cortes sin fractura visible,
por lo que se agrega con la media de los *k* logits más altos (top-k) en lugar de la media de todos.

**Modelos secuenciales.** Una GRU (Cho et al., 2014) bidireccional procesa la secuencia de
embeddings de cortes ordenados por z en ambos sentidos, pudiendo capturar contexto anatómico
entre cortes vecinos.

**Evaluación.** ROC-AUC resume la discriminación independiente del umbral (Hanley & McNeil, 1982);
el umbral operativo se fija con el índice de Youden J = sensibilidad + especificidad − 1
(Youden, 1950) en validación. La incertidumbre del AUC se estima con bootstrap percentil
(Efron & Tibshirani, 1993).

# 5. Revisión bibliográfica

- **Dataset.** Lin et al. (2023) describen el conjunto RSNA de fracturas cervicales en TC: estudios
  de múltiples instituciones, etiquetas por vértebra C1–C7 y anotaciones de cajas en un subconjunto.
  El EDA del Proyecto 2 confirmó en los CSV públicos: 2,019 estudios, 47.6 % positivos, cajas en 235
  estudios (todos positivos).
- **Reglas clínicas.** NEXUS (Hoffman et al., 2000) y la Canadian C-Spine Rule (Stiell et al., 2001)
  orientan cuándo solicitar imagen; el problema abordado aquí es posterior: interpretar la TC.
- **Arquitecturas.** Las soluciones de la competencia combinaron típicamente segmentación de
  vértebras, clasificación 2D/2.5D por corte y agregación secuencial; este trabajo no replica esos
  enfoques ni compara sus métricas (que además usan una métrica ponderada distinta sobre un test
  oculto), sino que implementa versiones mínimas de dos ideas: agregación MIL y agregación secuencial.

# 6. Metodología

## 6.1 Descripción del dataset

Se usan `train.csv` (2,019 estudios; `patient_overall` y C1–C7) y las imágenes DICOM de
`train_images/` de la competencia, obtenidas mediante la API oficial de Kaggle tras aceptar las reglas.
Los resultados del EDA tabular (Proyecto 2) se mantienen: datos sin nulos ni duplicados,
`patient_overall = 1` si y solo si alguna vértebra es positiva, mayor frecuencia de fractura en C7
(19.5 %), C2 (14.1 %) y C6 (13.7 %), y correlación entre vértebras vecinas (máx. C5–C6 = 0.36).

## 6.2 Muestreo y particiones

- 160 estudios muestreados de `train.csv`, estratificados por `patient_overall` (semilla 42).
- 80 % desarrollo / 20 % prueba; desarrollo → 80 % entrenamiento / 20 % validación; todo
  estratificado y **por estudio**. En RSNA 2022 cada estudio corresponde a un paciente distinto,
  por lo que no hay pacientes compartidos entre particiones (verificado: UIDs únicos).
- Particiones: entrenamiento 102 (49 +), validación 26 (12 +), prueba 32 (15 +)
  (`results/splits_summary.csv`).
- Por estudio se seleccionan **24 cortes** con una regla determinista: números de instancia
  equiespaciados entre el 5 % y el 95 % del total N (N obtenido por búsqueda binaria en Kaggle).
  **Este muestreo no garantiza incluir los cortes donde la fractura es visible.**

## 6.3 Obtención de datos

Descarga archivo por archivo vía API de Kaggle (sin descargar los ~340 GB). Durante la ejecución,
Kaggle respondió **HTTP 429** con `Retry-After ≈ 3800 s` tras ~600 peticiones en ~10 minutos
(cuota por ventana de tiempo). El descargador se rediseñó (v2) con limitador global, respeto exacto
de `Retry-After`, registro por estudio, reanudación sin duplicados y orden prueba → validación →
entrenamiento (sin modificar particiones). **[PENDIENTE: estudios finalmente descargados y tiempo total —
ver `results/download_log.csv` y ESTADO_FINAL.md]**

## 6.4 Preprocesamiento DICOM

| Paso | Implementación |
|---|---|
| Lectura | `pydicom.dcmread`; decodificación JPEG Lossless con pylibjpeg/gdcm |
| Intensidades | HU = pixel × RescaleSlope + RescaleIntercept, leídos por archivo |
| Padding | `PixelPaddingValue` → −1024 HU (aire) si está presente |
| Fotometría | MONOCHROME1 invertido; MONOCHROME2 sin cambios |
| Ventana | ósea L400/W1800 → [−500, 1300] HU |
| Normalización | lineal a [0, 1]; en la red, canal replicado ×3 y normalización ImageNet |
| Tamaño | 224×224, `cv2.INTER_AREA` (sin remuestreo isotrópico) |
| Orden | por `ImagePositionPatient[2]` (z), no por número de instancia |

Hallazgos de la auditoría sobre los primeros estudios procesados (`results/dicom_metadata.csv`,
`results/orden_z_auditoria.csv`):

- Tres codificaciones: Explicit VR Little Endian, Implicit VR Little Endian y JPEG Lossless (Process 14).
- RescaleIntercept −1024 (píxeles sin signo) en la mayoría y 0 con píxeles con signo en al menos un
  estudio; ambos producen HU coherentes (aire ≈ −1000, tejido blando ≈ 0).
- Grosor de corte variable (p. ej. 0.625 mm y 1.0 mm) y espaciado de píxel variable (≈0.35–0.43 mm).
- La dirección de z respecto al número de instancia **no es constante** (decreciente en la mayoría,
  creciente en algunos), lo que justifica ordenar por `ImagePositionPatient`.
- Fuera del campo de visión aparecen valores −2048 HU sin `PixelPaddingValue`; quedan en negro tras la ventana.

**[PENDIENTE: actualizar con la auditoría de todos los estudios procesados]**

## 6.5 Arquitecturas

- **Modelo A — ResNet18 + MIL.** ResNet18 (ImageNet) aplicada a cada uno de los 24 cortes →
  logit por corte → logit de estudio = media de los 3 logits más altos. Etiqueta: `patient_overall`.
- **Modelo B — ResNet18 + BiGRU.** Embedding de 512-d por corte → secuencia ordenada por z →
  GRU bidireccional (128 por dirección) → concatenación de media y máximo temporal → logit de estudio.

## 6.6 Configuración de entrenamiento

Idéntica para ambos modelos y fijada a priori: AdamW (lr 1e-4 backbone, 1e-3 cabeza, weight decay
1e-4), cosine annealing, 12 épocas, batch de 4 estudios (96 imágenes) con acumulación de gradientes ×2,
precisión mixta (AMP fp16), aumento consistente por estudio (flip horizontal, desplazamiento ±12 px),
`pos_weight = negativos/positivos` calculado **solo con entrenamiento**, semilla 42. Si ocurre OOM, el
batch se reduce a la mitad duplicando la acumulación. Uso de VRAM medido: ≈1.5 GB pico.

## 6.7 Selección y métricas

La época se elige por ROC-AUC de **validación**; el umbral, por índice de Youden en **validación**.
El conjunto de prueba se evalúa una sola vez. Métricas: matriz de confusión, sensibilidad,
especificidad, precisión, F1, exactitud, ROC-AUC (IC95 % bootstrap, 2000 remuestreos), tiempo de
entrenamiento y de inferencia por estudio.

# 7. Resultados experimentales

> Datos reproducidos de forma independiente por Hansel López (P3) en `notebooks/03_evaluacion.ipynb`,
> a partir de `artifacts/v2/results/`. Métricas recalculadas coinciden con las reportadas en
> `metrics_*.json` con diferencia < 1e-6.

## 7.1 Tamaño de muestra y particiones

Particiones disjuntas por `StudyInstanceUID` (verificado), estratificadas por `patient_overall`, semilla 42.

| Partición | Estudios | Con fractura | Prevalencia |
|-----------|----------|--------------|-------------|
| Entrenamiento | 1292 | 615 | 47.6 % |
| Validación | 323 | 154 | 47.7 % |
| Prueba | 404 | 192 | 47.5 % |

Prevalencia balanceada entre particiones (47.5–47.7 %).

## 7.2 Métricas en prueba (umbral fijado en validación — índice de Youden)

| Métrica | MIL (top-3) | BiGRU | Δ (BiGRU − MIL) |
|---------|-------------|-------|------------------|
| ROC-AUC | 0.638 [0.583, 0.693] | **0.737** [0.688, 0.784] | **+0.099** |
| Sensibilidad | 0.365 | **0.641** | +0.276 |
| Especificidad | **0.816** | 0.651 | −0.165 |
| Precisión | **0.642** | 0.624 | −0.018 |
| F1 | 0.465 | **0.632** | +0.167 |
| Exactitud | 0.601 | **0.646** | +0.045 |
| TP / FP / TN / FN | 70 / 39 / 173 / 122 | 123 / 74 / 138 / 69 | — |

IC95 % bootstrap (2000 remuestreos, semilla 42). Los IC95 % apenas se solapan en AUC
(0.693 vs 0.688), lo que sugiere —y las pruebas pareadas confirman— una diferencia real.

## 7.3 Visualizaciones

Figuras generadas en `artifacts/v2/results/figures/` y reproducidas en `notebooks/03_evaluacion.ipynb`:

- `comparacion_roc_prueba.png` — curvas ROC superpuestas con AUC e IC95 %.
- `comparacion_metricas_prueba.png` — barras comparativas de todas las métricas.
- `comparacion_probabilidades_prueba.png` — distribución de probabilidades por clase real.
- `comparacion_auc_validacion.png` — selección de época por AUC de validación.
- `comparacion_tiempos.png` — entrenamiento e inferencia.
- `confusion_{mil,bigru}.png` — matrices de confusión.
- `roc_{mil,bigru}.png` — curvas ROC individuales.
- `curvas_{mil,bigru}.png` — pérdida y AUC por época.

## 7.4 Tiempos de ejecución (Tesla T4, Kaggle)

| Modelo | Entrenamiento | s/época | Inferencia (GPU) | VRAM pico |
|--------|---------------|---------|------------------|-----------|
| MIL | 13.7 min | ~81 s | 28.0 ms/estudio | 3.21 GB |
| BiGRU | 12.4 min | ~81 s | 23.8 ms/estudio | 3.21 GB |

La inferencia excluye lectura y preprocesamiento DICOM (~2.5 s/estudio en CPU), que dominan
el tiempo de extremo a extremo.

# 8. Comparación de modelos y selección final

## 8.1 Pruebas estadísticas pareadas (mismo conjunto de prueba, n = 404)

| Prueba | Resultado | Interpretación |
|--------|-----------|----------------|
| **DeLong** (1988) | ΔAUC = +0.099, z = 4.09, p = 4.3·10⁻⁵ | Diferencia **altamente significativa** |
| **Bootstrap pareado** (2000 remuestreos) | IC95 % ΔAUC: [+0.048, +0.147]; 0 % de remuestreos ≤ 0 | BiGRU consistentemente superior |
| **McNemar exacto** (umbral val) | 66 vs 48 aciertos exclusivos, p = 0.11 | Dirección favorable a BiGRU, no significativo en predicciones binarias |
| **Correlación Spearman** | ρ = 0.60 | Acuerdo moderado entre probabilidades |

En validación (n = 323), la dirección es la misma pero la diferencia no alcanza significación
(DeLong p = 0.115; bootstrap IC95 % [−0.010, +0.086]).

## 8.2 Selección: **B · ResNet18 + BiGRU**

**Criterio fijado antes de entrenar:** AUC de prueba y equilibrio sensibilidad/especificidad.

**Justificación:**

1. **AUC superior y estadísticamente significativo:** 0.737 vs 0.638 (DeLong p < 0.001).
2. **Mejor sensibilidad** (0.641 vs 0.365): detecta 53 más fracturas reales que MIL (123 vs 70 TP).
3. **Mejor F1** (0.632 vs 0.465): balance más equilibrado entre precisión y sensibilidad.
4. **Estabilidad entre validación y prueba:** AUC estable (0.725 → 0.737), mientras que MIL cayó (0.687 → 0.638).
5. **Inferencia más rápida:** 23.8 vs 28.0 ms/estudio.

**Matiz importante:** la McNemar no alcanza significación en predicciones binarias (p = 0.11),
lo que indica que la diferencia es más clara en probabilidades continuas (DeLong) que en
clasificaciones binarias con un umbral fijo. Con los umbrales de validación, MIL opera con
baja sensibilidad (0.365) y BiGRU con un punto más equilibrado (0.641 / 0.651).

# 9. Aplicación desarrollada

`experimento_dl/app_streamlit.py` (Streamlit + Plotly), seis secciones: (1) dataset y EDA
interactivo, (2) preprocesamiento con ejemplos antes/después, montajes y auditoría, (3) predicción
sobre un estudio DICOM subido o un estudio de prueba en caché, con selección de modelo o comparación
de ambos, (4) métricas, (5) comparación de modelos y (6) limitaciones. Los estudios subidos pasan por
las mismas funciones `slice_record`, `build_volume` y `select_uploaded` usadas para entrenar
(verificado: volumen idéntico al de entrenamiento). Si no hay modelos entrenados, la app lo indica y no
muestra predicciones. Las etiquetas reales se muestran separadas y rotuladas como tales.

# 10. Limitaciones

- Muestra pequeña y una sola partición/semilla: alta varianza de las métricas (ver IC95 %).
- 24 cortes por estudio: la fractura puede no estar en los cortes muestreados (ruido de etiqueta).
- Clasificador global: no localiza la fractura ni identifica la vértebra.
- Sin segmentación de vértebras ni recorte de la región cervical: parte de los cortes cae en cráneo o tórax.
- Sin remuestreo isotrópico: el tamaño físico del píxel varía entre estudios.
- Cuota de descarga de Kaggle: condiciona el número de estudios efectivamente disponibles.
- Sin validación clínica ni externa; las métricas no son comparables con las de la competencia.

# 11. Conclusiones

El experimento completó exitosamente el pipeline de clasificación de fracturas cervicales en TC
desde el preprocesamiento DICOM hasta la evaluación comparativa de modelos, utilizando los 2019
estudios completos del dataset RSNA 2022 y entrenamiento en GPU Tesla T4 mediante Kaggle Notebooks.

**Hallazgos principales:**

1. **BiGRU supera estadísticamente a MIL** (ΔAUC = +0.099, p < 0.001 según prueba de DeLong).
   El bootstrap pareado confirma que en el 100% de los remuestreos BiGRU tuvo mejor AUC, con IC95%
   de la diferencia [+0.048, +0.147] que no incluye el cero.

2. **BiGRU ofrece mejor sensibilidad** (0.641 vs 0.365), detectando 53 fracturas reales más que MIL
   (123 vs 70 TP). Esto es clínicamente relevante: en screening médico, los falsos negativos son más
   costosos que los falsos positivos.

3. **Ambos modelos tienen desempeño moderado.** El mejor AUC alcanzado (0.737) está lejos del umbral
   de utilidad clínica (>0.90). La especificidad de BiGRU (0.651) genera 74 falsos positivos, lo que
   limitaría su uso como herramienta autónoma.

4. **El modelado secuencial captura contexto anatómico.** BiGRU procesa los 48 cortes ordenados por z,
   permitiendo que la red aprenda relaciones espaciales entre vértebras adyacentes. MIL trata cada corte
   independientemente y solo promedia los top-3, perdiendo información contextual.

**Limitaciones críticas:**

- **Una sola partición train/val/test** (semilla 42): los resultados pueden no generalizar a otras
  divisiones de datos. Se requiere validación cruzada k-fold para estimar varianza.
- **Muestreo de 48 cortes** de estudios con 69-1082 cortes totales: muchas fracturas pueden no estar
  en los cortes seleccionados, introduciendo ruido de etiqueta.
- **Prevalencia del 47.5%** en el conjunto de prueba es artificialmente alta comparada con la población
  clínica real (~5-10% en pacientes de trauma), lo que infla métricas como precisión.
- **Sin validación externa:** los modelos se evaluaron solo en datos de las mismas instituciones del
  entrenamiento. No se conoce el desempeño en poblaciones demográficas o equipos de TC diferentes.

**Recomendaciones:**

1. **No desplegar en producción clínica** sin validación prospectiva con radiólogos y datos de
   instituciones independientes.
2. **Incrementar el muestreo de cortes** (ej. 96 o 128 posiciones) para reducir el riesgo de no
   muestrear la fractura.
3. **Implementar validación cruzada 5-fold** para obtener estimaciones más robustas del desempeño y
   su varianza.
4. **Explorar arquitecturas 3D** (ej. 3D ResNet) que procesen el volumen completo sin necesidad de
   muestreo de cortes.
5. **Ajustar el umbral de decisión** según el costo clínico de falsos positivos vs falsos negativos,
   en consulta con radiólogos.

**Contribución académica:**

Este proyecto demuestra que es posible construir un pipeline reproducible de clasificación médica en
TC con recursos computacionales limitados (GPU de 16 GB, ~26 min de entrenamiento), utilizando
herramientas de código abierto (PyTorch, pydicom, scikit-learn) y prácticas de ciencia de datos
rigurosas (particiones estratificadas, validación independiente, pruebas estadísticas pareadas).
El código, los artefactos y el notebook de evaluación están versionados en GitHub y son completamente
reproducibles por terceros.

# Referencias

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).
Learning phrase representations using RNN encoder–decoder for statistical machine translation. In
*Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*
(pp. 1724–1734).

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale
hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*
(pp. 248–255).

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.

Flanders, A., et al. (2022). *RSNA 2022 Cervical Spine Fracture Detection*
[Competencia]. Kaggle. https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection

Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating
characteristic (ROC) curve. *Radiology, 143*(1), 29–36.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778).

Hoffman, J. R., Mower, W. R., Wolfson, A. B., Todd, K. H., & Zucker, M. I. (2000). Validity of a set of
clinical criteria to rule out injury to the cervical spine in patients with blunt trauma.
*New England Journal of Medicine, 343*(2), 94–99.

Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning. In
*Proceedings of the 35th International Conference on Machine Learning* (pp. 2127–2136). PMLR.

Lin, H. M., Colak, E., Richards, T., Kitamura, F. C., Prevedello, L. M., Talbott, J., … Flanders, A. E.
(2023). The RSNA cervical spine fracture CT dataset. *Radiology: Artificial Intelligence, 5*(5), e230034.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch:
An imperative style, high-performance deep learning library. In *Advances in Neural Information
Processing Systems 32* (pp. 8024–8035).

Stiell, I. G., Wells, G. A., Vandemheen, K. L., Clement, C. M., Lesiuk, H., De Maio, V. J., … Worthington,
J. R. (2001). The Canadian C-spine rule for radiography in alert and stable trauma patients.
*JAMA, 286*(15), 1841–1848.

Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer, 3*(1), 32–35.

> Nota para el equipo: completar la lista de autores de Lin et al. (2023) y de la ficha de Kaggle
> (Flanders et al., 2022) desde las fuentes originales antes de la entrega en PDF (aquí se abreviaron
> para no introducir nombres sin verificar).
