# Ejecutar el entrenamiento v2 en Kaggle — guía paso a paso

Notebook: `rsna_entrenamiento_v2.ipynb` (generado por `build_notebook.py`; no editar a mano).
Contexto, auditoría y decisión de muestreo: `../../docs/KAGGLE_NOTEBOOK.md`.

## 0. Requisitos de la cuenta
- Reglas de la competencia **RSNA 2022 Cervical Spine Fracture Detection** aceptadas.
- Teléfono verificado (necesario para GPU e Internet en notebooks).

## 1. Obtener el código en Kaggle (elegir una vía)

**Vía A — GitHub (por defecto).** Requiere que la rama `feature/entrenamiento` esté publicada en
`github.com/Paul-1511/Proyecto2-DataScience`. El notebook la clona (`BRANCH`), con Internet activado.

**Vía B — Kaggle Dataset (no depende de GitHub).**
1. En local: `python kaggle/make_code_bundle.py` → `data/kaggle_bundle/rsna_codigo.zip`
   (código del commit actual vía `git archive`; incluye `VERSION.txt` con el hash).
2. Kaggle → *Datasets* → *New Dataset* → subir el zip (privado), p. ej. slug `rsna-codigo-v2`.
3. En el notebook: *Add Input* → ese dataset y fijar
   `CODE_DATASET = "/kaggle/input/rsna-codigo-v2/rsna_codigo"`.

## 2. Crear y configurar el notebook
1. *Code* → *New Notebook* → *File* → *Import Notebook* → `rsna_entrenamiento_v2.ipynb`.
2. *Add Input* → competición RSNA 2022 (aparece en `/kaggle/input/rsna-2022-cervical-spine-fracture-detection`).
3. *Settings* → *Accelerator*: **GPU T4 x2** (o T4) · *Internet*: **On** · *Persistence*: opcional.
4. (Opcional, API) en vez de importar a mano: copiar `kernel-metadata.template.json` como
   `kernel-metadata.json`, poner el usuario y ejecutar `kaggle kernels push -p <carpeta>`.

## 3. Ejecutar
1. Interactivamente, las celdas 2–6 (≈5 min): verifican GPU (bloquea P100 si no está soportada),
   datos montados, decodificación JPEG Lossless, código y `train.csv` idéntico al de Kaggle.
2. *Save Version* → **Save & Run All (Commit)**. Estimado 3–5 h (preprocesamiento 20–40 min;
   ~6–9 min/época/modelo; máx. 15 épocas con early stopping de paciencia 4).
3. Revisar en el log: estudios con error de preprocesamiento, épocas elegidas, `qc_report.json` → `all_ok`.

## 4. Recuperar artefactos (pestaña *Output* de la versión)
- `weights/mil_final.pt`, `weights/bigru_final.pt` (~45 MB c/u) → `experimento_dl/data/weights/` (no se versionan).
- `results_v2.zip` → descomprimir en `experimento_dl/results_v2/` (sí se versiona: métricas, predicciones, figuras).
- `cache_v2/` (~5–6 GB): solo para reentrenar sin repreprocesar (`CACHE_INPUT`).

Después, en local: `RSNA_EXP=v2 python s07_qc.py --n-infer 0` (re-verifica métricas contra predicciones)
y completar `docs/RESULTADOS_ENTRENAMIENTO.md` con los valores de `results_v2/`.

## Fallos previstos y qué hacer
| Síntoma | Causa probable | Acción |
|---|---|---|
| Assert "GPU no soportada" | P100 con PyTorch sin `sm_60` | Accelerator → T4 |
| `git clone` falla | Internet apagado o rama no publicada | Activar Internet o usar la vía B |
| Error al decodificar JPEG | faltan pylibjpeg/gdcm (sin Internet) | Activar Internet |
| Pesos ImageNet no descargan | sin Internet | Añadir un input con `resnet18*.pth` (se detecta solo) |
| OOM | GPU pequeña | `s04` reduce el batch automáticamente (mantiene el efectivo) |
| Sesión > 12 h | GPU lenta | reanudar: `CACHE_INPUT` + `MODELS` restantes |
