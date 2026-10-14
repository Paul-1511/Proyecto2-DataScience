# Informe de calidad DICOM — RSNA 2022 (generado por `scripts/eda_metadatos_dicom.py`)

- Fuente: **auditoría del pipeline (artifacts/v2/results/dicom_metadata.csv.gz)**
- Estudios analizados: **2019** (semilla 42; todos los estudios de la partición v2)
- Cortes leídos: 96912 (48 centrales por estudio)

## Estadística descriptiva (por estudio)

|       |   n_slices |   thickness |   pixel_mm |   spacing_z |
|:------|-----------:|------------:|-----------:|------------:|
| count |   2019     |    2019     |   2019     |    2019     |
| mean  |    352.452 |       0.731 |      0.328 |       0.626 |
| std   |    132.01  |       0.185 |      0.078 |       0.204 |
| min   |     69     |       0.488 |      0.19  |       0.312 |
| 25%   |    257     |       0.625 |      0.271 |       0.45  |
| 50%   |    314     |       0.625 |      0.312 |       0.625 |
| 75%   |    429     |       1     |      0.355 |       0.7   |
| max   |   1082     |       1     |      0.713 |       2.626 |

## Comprobaciones de calidad

| Comprobación | Estudios | % | Ejemplos (UID) |
|---|---:|---:|---|
| Matriz distinta de 512×512 | 3 | 0.15 | 22678, 23400, 8858 |
| RescaleIntercept distinto de −1024 y 0 (p. ej. −8192/−10240) | 144 | 7.13 | 10041, 1016, 10579 |
| RescaleSlope distinto de 1 | 141 | 6.98 | 10041, 1016, 10579 |
| Dimensiones de matriz distintas dentro del estudio | 0 | 0.0 | — |
| Slope/intercept distintos dentro del estudio | 0 | 0.0 | — |
| SliceThickness variable dentro del estudio | 0 | 0.0 | — |
| |SliceThickness − espaciado z medido| > 0.05 mm | 879 | 43.54 | 10014, 10016, 10058 |
| BitsStored fuera de {12,16} | 0 | 0.0 | — |
| PhotometricInterpretation ≠ MONOCHROME2 | 0 | 0.0 | — |
| Atípico (IQR×1.5) en n.º de cortes | 27 | 1.34 | 1371, 15000, 15149 |
| Atípico (IQR×1.5) en SliceThickness | 0 | 0.0 | — |
| Atípico (IQR×1.5) en PixelSpacing | 111 | 5.5 | 10261, 10431, 11300 |
| Atípico (IQR×1.5) en espaciado z | 2 | 0.1 | 24673, 8693 |

## Limitaciones

- Solo 48 cortes centrales por estudio: no detecta huecos en z ni cabeceras fuera de esos cortes; n_slices proviene de n_files_study. En esta fuente spacing_z es la distancia entre dos cortes consecutivos centrales (sample_25d), no una mediana del estudio: la comprobación grosor-vs-espaciado z es orientativa y debe confirmarse con --dicom-dir.
- Comprobaciones omitidas por no existir la columna en la fuente: pixel_aniso, z_gap_ratio, z_dup, z_range_mm, manufacturer, kvp, kernel, non_axial.
