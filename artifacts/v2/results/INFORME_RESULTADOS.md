# Resultados del experimento (generado automáticamente por s05_compare_report.py)

Todos los valores provienen de `results/*.json|csv` producidos en esta ejecución.

## Datos

- Configuración: v2. Muestra planificada: 2019 estudios estratificados por `patient_overall` (semilla 42).
- Fuente: dataset montado en Kaggle; preprocesados: 2019.
- Posiciones por estudio: 48, equiespaciadas entre el 5% y 95% del rango de instancias; entrada 2.5D (vecinos a ±2 mm).

| partición | estudios | positivos |
|---|---|---|
| test | 404 | 192 |
| train | 1292 | 615 |
| val | 323 | 154 |

## Preprocesamiento (auditoría)

- Estudios con z únicos: 2019/2019; z monótono respecto a la instancia: 2018/2019.
- Dirección z vs. instancia: {'decreciente': 1821, 'creciente': 197, 'mixto': 1}.
- RescaleSlope: {1.0: 90144, 10.0: 6768}; RescaleIntercept: {-1024.0: 70992, 0.0: 19008, -10240.0: 6768, -8192.0: 144}.
- Transfer syntax: {'Explicit VR Little Endian': 59616, 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])': 22128, 'Implicit VR Little Endian': 15072, 'JPEG 2000 Image Compression': 96}.

## Modelos (prueba: estudios nunca usados para entrenar ni para elegir época/umbral)

| modelo                 | conjunto            |   n |   umbral |   roc_auc |   sensitivity |   specificity |   precision |    f1 |   accuracy |   TP |   FP |   TN |   FN | AUC_IC95_bootstrap   |   tiempo_entrenamiento_s |   inferencia_ms_estudio |   mejor_epoca_val |
|:-----------------------|:--------------------|----:|---------:|----------:|--------------:|--------------:|------------:|------:|-----------:|-----:|-----:|-----:|-----:|:---------------------|-------------------------:|------------------------:|------------------:|
| ResNet18 + MIL (top-3) | validación          | 323 |    0.55  |     0.687 |         0.481 |         0.811 |       0.698 | 0.569 |      0.653 |   74 |   32 |  137 |   80 | nan                  |                    nan   |                   nan   |               nan |
| ResNet18 + MIL (top-3) | prueba (umbral val) | 404 |    0.55  |     0.638 |         0.365 |         0.816 |       0.642 | 0.465 |      0.601 |   70 |   39 |  173 |  122 | [0.583, 0.693]       |                    822.7 |                    28   |                 6 |
| ResNet18 + MIL (top-3) | prueba (umbral 0.5) | 404 |    0.5   |     0.638 |         0.516 |         0.708 |       0.615 | 0.561 |      0.616 |   99 |   62 |  150 |   93 | nan                  |                    nan   |                   nan   |               nan |
| ResNet18 + BiGRU       | validación          | 323 |    0.548 |     0.725 |         0.721 |         0.657 |       0.657 | 0.687 |      0.687 |  111 |   58 |  111 |   43 | nan                  |                    nan   |                   nan   |               nan |
| ResNet18 + BiGRU       | prueba (umbral val) | 404 |    0.548 |     0.737 |         0.641 |         0.651 |       0.624 | 0.632 |      0.646 |  123 |   74 |  138 |   69 | [0.688, 0.784]       |                    741.6 |                    23.8 |                 5 |
| ResNet18 + BiGRU       | prueba (umbral 0.5) | 404 |    0.5   |     0.737 |         0.714 |         0.59  |       0.612 | 0.659 |      0.649 |  137 |   87 |  125 |   55 | nan                  |                    nan   |                   nan   |               nan |

AUC prueba: MIL=0.638, BiGRU=0.737. Los IC95% bootstrap se solapan; con este tamaño de prueba la diferencia no es concluyente.
