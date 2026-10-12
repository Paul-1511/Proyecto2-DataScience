# Experimento mínimo: clasificación global de estudios TC (RSNA 2022)

Experimento académico reducido sobre el dataset de Kaggle **RSNA 2022 Cervical Spine
Fracture Detection**. Objetivo: un pipeline reproducible de preprocesamiento DICOM,
entrenamiento y evaluación sobre estudios no vistos. **No es un sistema de detección ni de
localización de fracturas y no tiene validación clínica.**

Los resultados numéricos están únicamente en `results/` (generados por los scripts) y en
`results/INFORME_RESULTADOS.md` (generado por `s05_compare_report.py`).

## Ejecutar la aplicación (demostración)

Desde la raíz del repositorio (Windows, Git Bash o PowerShell), con el entorno `.venv` ya creado:

```bash
.venv/Scripts/streamlit run experimento_dl/app_streamlit.py
```

Abre http://localhost:8501. Si `.venv` no existe, primero se instalan las dependencias de
`experimento_dl/requirements.txt` (ver "Reproducción"). El visor DICOM usa los estudios de
`experimento_dl/data/dicom/` (no versionados) o archivos `.dcm` subidos desde la propia app.
Estado actual del proyecto: ver `ESTADO_PRESENTACION.md`.

## Reproducción

```bash
python -m venv --system-site-packages .venv          # desde la raíz del repo
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install pydicom pylibjpeg pylibjpeg-libjpeg python-gdcm kaggle opencv-python streamlit tabulate
# Token de Kaggle en la variable de entorno KAGGLE_API_TOKEN y reglas de la competencia aceptadas
cd experimento_dl
python s01_sample_split.py        # muestra + particiones  -> results/splits.csv
python s02_download.py            # descarga parcial (24 cortes/estudio), reanudable
python s03_preprocess.py          # DICOM -> HU -> ventana ósea -> 224x224 (incremental)
python s04_train_eval.py --model mil
python s04_train_eval.py --model bigru
python s05_compare_report.py      # tabla comparativa, figuras e informe
streamlit run app_streamlit.py
```

`data/` (DICOM, caché y pesos) no se versiona.

## Diseño

**Muestra y particiones** (`s01`). 160 estudios de `train.csv` (2019) estratificados por
`patient_overall`; 80 % desarrollo / 20 % prueba y, dentro de desarrollo, 80 % entrenamiento /
20 % validación, todo estratificado y por `StudyInstanceUID` (semilla 42). En RSNA 2022 cada
estudio corresponde a un paciente distinto (no se publica otro identificador de paciente), por lo
que no hay estudios ni pacientes compartidos entre particiones. Las etiquetas solo se usan como
objetivo y para estratificar.

**Descarga** (`s02`). API oficial de Kaggle, archivo por archivo, sin descargar el dataset
completo (~340 GB). El número de cortes N de cada estudio se obtiene con búsqueda binaria sobre la
existencia de `train_images/<UID>/<n>.dcm`.

**Selección de cortes.** 24 números de instancia equiespaciados entre el 5 % y el 95 % de N.
Regla determinista y reproducible. **Este muestreo no garantiza incluir los cortes donde la
fractura es visible**: el estudio se etiqueta como positivo aunque ninguno de los 24 cortes muestre
la fractura, y en estudios largos parte de los cortes cae fuera de la columna cervical (cráneo,
tórax).

**Preprocesamiento** (`s03`). Lectura con pydicom; HU = pixel × RescaleSlope + RescaleIntercept
leídos de cada archivo (se registran en `results/dicom_metadata.csv`); PixelPaddingValue → aire;
MONOCHROME1 invertido; ventana ósea nivel 400 / ancho 1800 HU (rango [−500, 1300]) y normalización
a [0, 1]; redimensión a 224×224 (INTER_AREA, sin remuestreo isotrópico); orden por
`ImagePositionPatient[2]` (z), auditado en `results/orden_z_auditoria.csv`. En la red, el canal se
replica a 3 y se normaliza con media/desv. de ImageNet.

**Modelos** (`s04`). Etiqueta solo a nivel de estudio (`patient_overall`).
- *ResNet18 + MIL*: ResNet18 preentrenada en ImageNet produce un logit por corte; el logit del
  estudio es la **media de los 3 logits de corte más altos**. No se asume que todos los cortes de un
  estudio positivo contengan fractura.
- *ResNet18 + BiGRU*: embeddings de 512-d por corte, secuencia ordenada por z, GRU bidireccional
  (128 por dirección), pooling [media, máx] → logit del estudio.

Mismos hiperparámetros fijados a priori para ambos (AdamW, lr 1e-4 backbone / 1e-3 cabeza,
cosine, 12 épocas, batch 4 estudios, AMP, aumento: flip horizontal y desplazamiento ±12 px).

**Selección y evaluación.** Época elegida por ROC-AUC de validación; umbral elegido en validación
(índice de Youden). El conjunto de prueba se evalúa una única vez con ese modelo y umbral. Se
reportan matriz de confusión, sensibilidad, especificidad, precisión, F1, exactitud, ROC-AUC con
IC95 % bootstrap (2000 remuestreos) y tiempos.

## Limitaciones

- Muestra pequeña (≈32 estudios de prueba): las métricas tienen alta varianza; ver IC95 %.
- Una sola partición y una sola semilla; no hay validación cruzada.
- 24 cortes por estudio de cientos: información muy parcial; la fractura puede no estar muestreada.
- Clasificador global: no localiza la vértebra fracturada ni la fractura.
- Sin validación clínica ni externa; las métricas no son comparables con las de la competencia
  (que usa una pérdida ponderada por vértebra sobre el conjunto de test oculto).
