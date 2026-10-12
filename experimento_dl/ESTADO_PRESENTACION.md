# Estado para la presentación (12-oct-2026)

## Resumen honesto

- **Funciona**: pipeline DICOM real (descarga parcial autorizada, lectura, HU, ventana ósea,
  normalización, 224×224, orden por `ImagePositionPatient`) y app Streamlit de demostración.
- **No hay modelos entrenados ni métricas.** La descarga quedó limitada por la cuota de la API de
  Kaggle y solo se completaron estudios de entrenamiento: no existen estudios de validación ni de
  prueba descargados, por lo que una evaluación válida no es posible. **No se reporta ninguna métrica.**

## Datos efectivamente disponibles (`results/auditoria_estudios_disponibles.csv`)

| partición | planificados | completos y preprocesados | positivos | parciales (incompletos) |
|---|---|---|---|---|
| entrenamiento | 102 | 12 | 4 | 7 |
| validación | 26 | 0 | 0 | 0 |
| prueba | 32 | 0 | 0 | 6 |

- 12 estudios × 24 cortes = 288 cortes DICOM procesados sin fallos (~0.13 GB).
- Particiones originales intactas (`results/splits.csv`, 102/26/32, estratificadas por estudio).
- Los DICOM, la caché y los logs se conservan en `data/` (no versionado).

## Qué funciona (verificado)

| Componente | Verificación |
|---|---|
| `s01_sample_split.py` | particiones por estudio sin solapamiento (UIDs únicos) |
| `s02_download.py` | descarga parcial vía API de Kaggle; reanudable; maneja HTTP 429/Retry-After |
| `s03_preprocess.py` | 12/12 estudios, 0 fallos; refactor verificado con volúmenes idénticos |
| Auditoría de orden z | 12/12 con z únicos y monótonos; dirección respecto a la instancia variable (11 decrecientes, 1 creciente) |
| Metadatos | 3 codificaciones (Explicit/Implicit VR, JPEG Lossless); intercept −1024 y 0; grosor 0.5–1.0 mm; píxel 0.25–0.53 mm |
| App Streamlit | arranca sin errores; 0 excepciones en todas las pestañas (comprobado en navegador) |
| Preprocesamiento de la app | mismo código que el pipeline; volumen idéntico al de entrenamiento con archivos en orden aleatorio |
| `s04`/`s05` | probados solo con una prueba de humo técnica en el scratchpad (sin valor como resultado; no incluida en `results/`) |

## App (`app_streamlit.py`)

1. **Dataset y EDA**: métricas de `train.csv`, gráficos interactivos (Plotly) y figuras EDA de `imgs/`.
2. **Preprocesamiento**: pasos documentados, figuras antes/después, montajes, auditoría z y metadatos.
3. **Visor DICOM**: estudio local o `.dcm` subidos → antes (HU crudo) / después (ventana ósea 224×224),
   histograma HU con la ventana, metadatos por corte, comparación de cortes y montaje.
4. **Modelos y predicción**: indica explícitamente **PENDIENTE (no entrenado)** para ambos modelos;
   no muestra predicciones simuladas.
5–6. **Métricas / Comparación**: indican que no hay resultados disponibles.
7. **Limitaciones**.

Ejecutar: `.venv/Scripts/streamlit run experimento_dl/app_streamlit.py` (desde la raíz del repo).

## Figuras reales generadas (`results/figures/`)

- `presentacion_antes_despues_multiestudio.png` — 6 estudios, HU crudo vs ventana ósea 224×224.
- `presentacion_variabilidad_metadatos.png` — grosor de corte, tamaño de píxel y codificación.
- `preproc_antes_despues_y{0,1}_*.png` — HU, histograma con ventana y resultado (4 estudios).
- `montaje_cortes_y{0,1}_*.png` — 24 cortes ordenados por z (4 estudios).

## Pendiente

- Descargar los estudios de validación (26) y prueba (32), y completar entrenamiento (90 restantes).
  Con la cuota observada (~8–9 peticiones/min, ~36 peticiones por estudio) son ~10 h; alternativa
  recomendada: ejecutar el pipeline en un Kaggle Notebook (`KAGGLE_NOTEBOOK.md`), donde los datos ya
  están montados.
- Entrenar y evaluar ResNet18 + MIL y ResNet18 + BiGRU (`run_experiment.py`), y generar la comparación.
- Completar las secciones [PENDIENTE] de `INFORME_TECNICO.md`.
- Diagnóstico abierto: en la última ejecución hubo errores no-429 durante la descarga (causa no
  identificada; se añadió registro detallado, pero la descarga se detuvo antes de capturarla).

## Limitaciones

- Sin modelo entrenado: la app no clasifica fracturas en esta versión.
- 12 estudios descargados (todos de entrenamiento) no permiten conclusiones sobre el desempeño.
- El diseño previsto (24 cortes/estudio, clasificador global) no localiza fracturas, y el muestreo no
  garantiza incluir los cortes donde la fractura es visible.
- No es una herramienta clínica ni tiene validación clínica.

## Qué se puede presentar honestamente

- El problema, el EDA tabular del Proyecto 2 y el diseño experimental (muestra, particiones sin fuga,
  selección de época/umbral solo en validación).
- El pipeline DICOM funcionando sobre datos reales, con la auditoría de metadatos y orden z.
- La app como demostración de exploración y preprocesamiento.
- El obstáculo real (cuota de Kaggle, HTTP 429) y cómo se abordó.
- **No** se pueden presentar métricas de clasificación, comparación entre modelos ni selección final.
