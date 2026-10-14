# Sección del informe — Aplicación interactiva

*(Para integrar por Méndez en el informe. Las figuras están en `docs/app/capturas/`.)*

## Aplicación interactiva

Se desarrolló una aplicación web con Streamlit que integra la exploración de datos, el preprocesamiento,
la predicción sobre estudios nuevos y la comparación de los dos modelos entrenados (A: ResNet18 + MIL;
B: ResNet18 + BiGRU). La aplicación no reimplementa el preprocesamiento ni las arquitecturas: usa el
módulo de inferencia del pipeline (`inference.py`), que reconstruye la configuración guardada con cada
checkpoint (48 posiciones 2.5D, ventana ósea L400/W1800, 224 × 224, orden por `ImagePositionPatient`).
Así, la entrada que recibe el modelo en la aplicación es la misma que en entrenamiento y evaluación;
esta compatibilidad se verificó en el notebook `04_integracion_app.ipynb` (forma, tipo, parámetros e
invariancia al orden de los archivos).

La aplicación se organiza en ocho pestañas (Figura X): (1) exploración del conjunto de datos y de los
metadatos DICOM; (2) descripción y ejemplos del preprocesamiento; (3) visor DICOM, que muestra cada corte
en unidades Hounsfield y tras la ventana ósea, su histograma, los metadatos y el tensor de entrada del
modelo; (4) predicción, donde el usuario sube los archivos DICOM de un estudio y elige el Modelo A, el
Modelo B o ambos, ya sea para un estudio o para un lote de estudios; (5) métricas por modelo y conjunto; (6) comparación interactiva; (7) estado de los
artefactos, y (8) limitaciones. Un panel lateral permite alternar entre gráficas interactivas (Plotly) y
figuras estáticas, y mostrar u ocultar cada tipo de gráfica.

En la pestaña de comparación, el usuario selecciona conjuntos y métricas; la aplicación presenta barras
agrupadas, la diferencia B − A, las pruebas pareadas (DeLong, bootstrap y McNemar), las curvas ROC con el
punto de operación, la distribución de probabilidades por clase y la dispersión de las probabilidades de
ambos modelos por estudio. Un control de umbral recalcula las métricas sobre las predicciones de prueba
guardadas, con intervalos de confianza bootstrap al 95 % calculados con el módulo de evaluación
independiente del equipo; se advierte que esta exploración es descriptiva, ya que los umbrales oficiales se fijaron en
validación. Todos los valores provienen de los artefactos del entrenamiento (`artifacts/v2/`); en ausencia
de un artefacto la aplicación muestra el estado PENDIENTE y nunca genera valores sustitutos.

Cuando se ejecutan ambos modelos sobre un mismo estudio, la aplicación compara sus salidas: indica si
coinciden en la clasificación, la diferencia de probabilidades y la posición de cada probabilidad dentro de
la distribución observada en el conjunto de prueba, separada por clase real. Si el estudio pertenece a las
particiones del experimento, la aplicación compara su probabilidad con la registrada durante la
evaluación, lo que permite verificar que el despliegue reproduce exactamente el modelo evaluado; si el
estudio pertenece a la partición de entrenamiento, se advierte que no constituye una evaluación
independiente. Los resultados pueden descargarse en CSV o JSON con el mismo formato que las predicciones
del experimento.

Para la predicción, cada resultado se acompaña del umbral elegido en validación, de los avisos de calidad
de la entrada (archivos ilegibles, estudios mezclados, menos cortes que posiciones del modelo, ausencia de
`RescaleSlope`) y de un descargo explícito: se trata de una clasificación global por estudio, sin
localización de la fractura y sin validación clínica. La etiqueta real, cuando se conoce, se muestra por
separado y rotulada como tal. Los checkpoints se verifican mediante SHA-256 antes de cargarse en modo
seguro (`weights_only=True`).

La aplicación se validó con 25 pruebas automáticas propias (carga y consistencia de métricas, manejo de
entradas inválidas, estado de checkpoints, forma de la entrada, comparación de modelos, flujo completo de
predicción en la interfaz e inferencia por lotes) y con una revisión en navegador de las ocho pestañas.
**[PENDIENTE: completar con `docs/app/validacion_inferencia.json` (reproducibilidad frente a la evaluación,
tiempo por estudio y acuerdo A/B) tras ejecutar `tools/validar_inferencia.py` con los checkpoints reales.
No afirmar que la predicción sobre nuevas entradas fue probada de extremo a extremo hasta entonces.]**

### Figuras sugeridas

| Figura | Archivo | Pie sugerido |
|---|---|---|
| X.1 | `capturas/01_datos_y_eda.png` | Exploración interactiva del conjunto de datos y metadatos DICOM. |
| X.2 | `capturas/03_visor_dicom.png` | Visor DICOM: corte en HU, ventana ósea y entrada 2.5D del modelo. |
| X.3 | `capturas/04_prediccion.png` | Selección de modelo y predicción (estado PENDIENTE sin checkpoints). |
| X.4 | `capturas/06_comparacion.png` | Comparación interactiva de los modelos A y B en el conjunto de prueba. |
