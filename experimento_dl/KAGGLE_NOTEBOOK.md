# Ejecución en Kaggle Notebooks (estrategia principal de entrenamiento)

La descarga local por API quedó descartada: Kaggle devolvió HTTP 429 con `Retry-After` creciente
(15 → 25 → 70 → 105 s; ~8–9 peticiones/min). En un Notebook de la competencia los DICOM están montados
en `/kaggle/input/` y no se descarga nada.

Notebook: [`kaggle/rsna_entrenamiento_v2.ipynb`](kaggle/rsna_entrenamiento_v2.ipynb)
(generado por `kaggle/build_notebook.py`; no contiene lógica propia: ejecuta los scripts versionados).

## Auditoría de la versión anterior de este documento (v1) y correcciones

| Problema en v1 | Corrección |
|---|---|
| Caché sin comprimir dentro del repo clonado | `RSNA_CACHE_DIR=/kaggle/working/cache_v2`, `.npz` comprimido (~2.8 MB/estudio) |
| `s04` cargaba todos los estudios en RAM | `StudyDataset` + `DataLoader` con lectura bajo demanda |
| `s03` secuencial | `ProcessPoolExecutor` (un proceso por núcleo), errores registrados por estudio |
| Pesos ImageNet solo por internet | Alternativa offline `RESNET18_WEIGHTS` (archivo en `/kaggle/input`) |
| Sin verificación de GPU | Comprobación de arquitectura (P100 `sm_60` puede no estar soportada) |
| Sin early stopping ni checkpoint autocontenido | `--patience`; `<modelo>_final.pt` con configuración y umbral |
| Muestra de 160 estudios (prueba n=32) | v2: 2019 estudios (prueba n=404), ver "Muestreo" |
| Sin control de calidad | `s07_qc.py` (particiones, métricas vs predicciones, inferencia desde DICOM) |

## Pasos (los realiza un miembro del equipo con cuenta de Kaggle)

1. Aceptar las reglas de la competencia con la cuenta que ejecutará el notebook.
2. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* → subir `rsna_entrenamiento_v2.ipynb`.
3. Panel *Settings*: *Add Input* → competición **RSNA 2022 Cervical Spine Fracture Detection**;
   *Accelerator* → **GPU T4 x2**; *Internet* → **On** (requiere verificación telefónica).
4. Ejecutar las primeras 4 celdas de forma interactiva (verifican GPU, datos, decodificación JPEG y código).
5. *Save Version* → **Save & Run All (Commit)**. Duración estimada: 3–5 h
   (preprocesamiento 20–40 min con 4 CPU; ~6–9 min por época por modelo en T4, máx. 15 épocas con
   early stopping). Límite de Kaggle: 12 h por sesión; cuota GPU semanal ~30 h.
6. Descargar de *Output*: `weights/mil_final.pt`, `weights/bigru_final.pt`, `results_v2.zip`.
   Para reentrenar sin repreprocesar: *Add Input* → Output de la versión anterior y
   `CACHE_INPUT = "/kaggle/input/<notebook>/cache_v2"`.

**Requisito:** el notebook clona la rama `feature/entrenamiento` del repositorio público; esa rama debe
estar publicada en GitHub antes de ejecutarlo (alternativa: subir `experimento_dl/` como Kaggle Dataset
y ajustar `CODE`).

## Muestreo de cortes: análisis y decisión (tomada antes de cualquier resultado)

**Limitaciones de v1 (24 cortes 2D equiespaciados sobre el 5–95 % del estudio):**
- Con N≈400 cortes de 0.625 mm, el paso entre cortes muestreados es ≈15 cortes ≈ 9–10 mm: una fractura
  visible en pocos milímetros puede quedar entre dos cortes.
- El estudio incluye cráneo y tórax; parte de las 24 posiciones cae fuera de C1–C7.
- Los 3 canales de ResNet se llenaban con la misma imagen (información redundante).
- La etiqueta es por estudio: si la fractura no aparece en los cortes, la etiqueta es ruidosa para el modelo.

**Alternativas evaluadas (todas sin usar etiquetas ni anotaciones del conjunto de prueba):**

| Alternativa | Cobertura | Coste | Decisión |
|---|---|---|---|
| Mantener 24 cortes 2D | baja (~10 mm entre cortes) | 1× | descartada como principal |
| Más cortes 2D (p. ej. 64) | media | ~2.7× GPU | posible, pero desaprovecha canales |
| **2.5D: 48 posiciones × (corte ± 2 mm)** | paso ≈5 mm, cada posición abarca ±2 mm | ~2× GPU que v1, 3× lecturas | **elegida (v2)** |
| Recorte de la región cervical con segmentación (TotalSegmentator / máscaras de la competencia) | alta en C1–C7 | modelo adicional; las máscaras solo existen en 87 estudios de entrenamiento | descartada: no aplicable a estudios nuevos sin una segmentación propia |
| Usar las bounding boxes para elegir cortes | — | — | descartada: solo 235 estudios positivos; sesgo y fuga hacia la etiqueta |

**v2:** 48 posiciones equiespaciadas (misma regla 5–95 %), cada una con 3 canales = cortes a −2 mm,
0 y +2 mm (desplazamiento en cortes = round(2 mm / espaciado z), mínimo 1), ordenados por z.
Mismo HU, ventana ósea L400/W1800, normalización y 224×224 que v1. Aplicable a cualquier estudio nuevo
(solo usa metadatos DICOM). Sigue sin garantizar que la fractura quede muestreada (limitación documentada).

**Coste medido localmente:** decodificación ~11 ms/corte (20 ms JPEG Lossless) → ~1.6 s/estudio;
entrenamiento K=48 2.5D: ~142 img/s y 2.8 GB de VRAM en RTX 3060 Laptop (cabe también en 6 GB).

## Particiones v2

`results_v2/splits.csv` — generado con el mismo `s01_sample_split.py` (semilla 42) sobre los 2019
estudios: 80/20 desarrollo/prueba y 80/20 entrenamiento/validación → **1292 / 323 / 404 (64/16/20)**,
prevalencia 0.476 en las tres. Las particiones v1 (`results/splits.csv`, 102/26/32) no se modifican
(se verificó que se regeneran idénticas). `s01` se niega a sobrescribir un archivo de particiones distinto.
Los estudios que fallen en el preprocesamiento se excluyen y se informan por partición (no se reasignan).
