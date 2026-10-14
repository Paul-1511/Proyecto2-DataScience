# Entrega P2 — EDA de imágenes e informe (rama feature/eda-imagenes)

Contenido (rutas relativas a la raíz del repo; reemplaza `p2_eda_imagenes/`):
- notebooks/02_eda_imagenes.ipynb  (+ notebooks/tools/build_notebook_02.py)
- scripts/eda_metadatos_dicom.py
- artifacts/eda_imagenes/auditoria_v2/  (línea base generada desde artifacts/v2, SIN Kaggle)
- informe/  (informe_final.tex, secciones/, referencias.bib, generated/, figs/, tools/gen_tablas_eda.py, informe_final.pdf)

Estado: el notebook NO se ha ejecutado en Kaggle (modo DICOM sin probar). En modo auditoría corre completo.

Kaggle: 1) pushear este layout a feature/eda-imagenes; 2) notebook nuevo, subir el .ipynb; 3) Input = competencia,
Internet ON; 4) Run All (N_STUDIES=300); 5) bajar /kaggle/working/eda_imagenes y correr
`python informe/tools/gen_tablas_eda.py artifacts/eda_imagenes` y recompilar.

Compilar informe: xelatex informe_final && bibtex informe_final && xelatex informe_final && xelatex informe_final
No se modificó s03_preprocess.py ni configuraciones de entrenamiento. No se hizo merge.
