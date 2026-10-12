# Alternativa: ejecutar el mismo pipeline en un Kaggle Notebook (GPU)

Motivo: la API de Kaggle impone una cuota de descargas por ventana de tiempo (HTTP 429 con
`Retry-After` ≈ 1 h tras ~600 peticiones). En un Notebook de la competencia, `train_images/`
está montado en `/kaggle/input/...` y no hay que descargar nada.

Garantías de equivalencia:
- Mismas particiones: se usa `results/splits.csv` del repositorio (no se vuelve a muestrear).
- Mismo muestreo de cortes: `s03` aplica `select_slice_numbers(N)` cuando el directorio del
  estudio contiene todos los cortes (N = número de instancia máximo), igual que `s02`.
- Mismo preprocesamiento, modelos, hiperparámetros y semillas.

Pasos (requiere que un miembro del equipo cree el notebook en su cuenta, con GPU e internet
activados para descargar los pesos de ResNet18):

```bash
# 1) En el notebook: añadir la competencia "RSNA 2022 Cervical Spine Fracture Detection" como input
# 2) Copiar el repositorio (git clone del repo público o subir experimento_dl/ como dataset)
pip install pydicom pylibjpeg pylibjpeg-libjpeg python-gdcm
cd experimento_dl
export RSNA_DICOM_DIR=/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images
python s03_preprocess.py      # procesa los estudios de splits.csv (24 cortes cada uno)
python s04_train_eval.py --model mil
python s04_train_eval.py --model bigru
python s05_compare_report.py
# 3) Descargar results/ (métricas, figuras) y data/weights/*.pt para la app local
```

Nota: `s03` en este modo procesa todos los estudios de `splits.csv` presentes en el dataset
montado; no utiliza ningún estudio fuera de la muestra.
