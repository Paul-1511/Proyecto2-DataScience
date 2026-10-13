"""Genera notebooks/02_eda_imagenes.ipynb (P2). Ejecutar: python notebooks/tools/build_notebook_02.py
El notebook NO contiene resultados escritos a mano: toda cifra se calcula al ejecutarlo."""
from pathlib import Path
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# 02 · EDA de imágenes DICOM — RSNA 2022 Cervical Spine Fracture Detection
**Proyecto final CC3084 Data Science (UVG) · Responsabilidad P2 · rama `feature/eda-imagenes`**

## Objetivo
Caracterizar la heterogeneidad **real** de los estudios de TC (parámetros de adquisición, dimensiones, número de cortes,
intensidades) y documentar la calidad de los metadatos DICOM, para justificar las decisiones de preprocesamiento
(HU → ventana ósea → 224×224, orden por z) y delimitar las limitaciones del experimento.

## Metodología
1. Muestra **reproducible** de estudios (semilla 42, estratificada por `patient_overall`) leída desde Kaggle.
2. Extracción de **cabeceras** DICOM de todos los cortes de cada estudio (sin leer píxeles) con `scripts/eda_metadatos_dicom.py`.
3. Estadística descriptiva por estudio, detección de atípicos (regla IQR×1.5) y comprobaciones de consistencia.
4. Análisis de **intensidades** (HU) en el corte central de una submuestra que incluye todos los protocolos atípicos hallados.
5. Comparación visual **antes/después** del preprocesamiento usando las funciones de `s03_preprocess.py` *sin modificarlas*.
6. Exportación de artefactos (CSV/JSON/PNG) para el informe y los demás integrantes.

> No se usan etiquetas para decidir nada del preprocesamiento; `patient_overall` solo se emplea para estratificar el muestreo y comparar distribuciones."""))

C.append(md("""## DEPENDENCIAS E INTEGRACIÓN
| # | Pregunta | Respuesta |
|---|---|---|
| 1 | **Archivos que necesita** | `train.csv` y `train_images/<StudyInstanceUID>/<n>.dcm` de la competencia; módulos `scripts/eda_metadatos_dicom.py` y (solo lectura) `experimento_dl/s03_preprocess.py`, `experimento_dl/common.py`. Opcional: `artifacts/v2/results/dicom_metadata.csv.gz` (modo auditoría). |
| 2 | **Quién los produce** | Competencia RSNA 2022 (Kaggle); código del repositorio (rama `main`); la auditoría `dicom_metadata.csv.gz` la produce la responsable GPU (Majo). |
| 3 | **Dónde se encuentran** | `/kaggle/input/rsna-2022-cervical-spine-fracture-detection/` (Kaggle) · repo clonado en `/kaggle/working/Proyecto2-DataScience`. |
| 4 | **Cómo se cargan** | Celda de *Configuración*: detecta Kaggle, clona la rama y añade `scripts/` y `experimento_dl/` a `sys.path` (importa módulos; **no** ejecuta otros notebooks). |
| 5 | **Si no existen** | Sin DICOM: el notebook usa la auditoría del pipeline (solo metadatos de 48 cortes centrales por estudio) y **omite** las secciones de píxeles indicándolo; si tampoco hay auditoría se detiene con un mensaje. Nunca genera datos de relleno. |
| 6 | **Qué entrega** | `artifacts/eda_imagenes/`: `metadata_studies.csv`, `metadata_slices.csv.gz` (solo con DICOM), `quality_report.json`, `INFORME_CALIDAD_DICOM.md`, `intensidades_corte_central.csv`, `figures/*.png`, `MANIFEST_EDA.json`. Consumidores: **Méndez/P2 (informe)**, **Hansel (comparación)**, **Cabrera (EDA interactivo de la app)**. |"""))

C.append(md("## 1. Configuración"))
C.append(code('''import os, sys, json, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ON_KAGGLE = Path("/kaggle/input").exists()
COMP = "rsna-2022-cervical-spine-fracture-detection"
BRANCH, REPO_URL = "feature/eda-imagenes", "https://github.com/Paul-1511/Proyecto2-DataScience.git"

if ON_KAGGLE:
    REPO = Path("/kaggle/working/Proyecto2-DataScience")
    if not REPO.exists():
        for br in (BRANCH, "main"):       # si la rama aún no está publicada se usa main y se avisa
            r = subprocess.run(["git", "clone", "--depth", "1", "-b", br, REPO_URL, str(REPO)])
            if r.returncode == 0:
                print("Código clonado desde la rama:", br); break
    DATA, OUT = Path(f"/kaggle/input/{COMP}"), Path("/kaggle/working/eda_imagenes")
else:
    REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    DATA, OUT = Path(os.environ.get("RSNA_COMP_DIR", "/nonexistent")), REPO / "artifacts" / "eda_imagenes"

DICOM_DIR = DATA / "train_images"
HAS_DICOM = DICOM_DIR.exists()
TRAIN_CSV = DATA / "train.csv" if (DATA / "train.csv").exists() else REPO / "train.csv"
AUDIT = REPO / "artifacts" / "v2" / "results" / "dicom_metadata.csv.gz"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# Los módulos de experimento_dl leen estas variables al importarse; se redirigen a un scratch (no se escribe en el repo).
SCRATCH = OUT / "_scratch"
os.environ.update(RSNA_EXP="v2", RSNA_DICOM_DIR=str(DICOM_DIR if HAS_DICOM else SCRATCH / "dicom_vacio"),
                  RSNA_CACHE_DIR=str(SCRATCH / "cache"), RSNA_WEIGHTS_DIR=str(SCRATCH / "w"), RSNA_RESULTS_DIR=str(SCRATCH / "res"))
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "experimento_dl")]

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
import eda_metadatos_dicom as E
pd.set_option("display.width", 160); plt.rcParams["figure.dpi"] = 100
print(f"Kaggle: {ON_KAGGLE} | DICOM disponibles: {HAS_DICOM} | train.csv: {TRAIN_CSV.exists()} | auditoría: {AUDIT.exists()}")
if not HAS_DICOM and not AUDIT.exists():
    raise SystemExit("No hay DICOM ni auditoría: añade el dataset de la competencia como entrada (Add Input).")'''))

C.append(md("""## 2. Datos utilizados y tamaño de muestra
Con DICOM: muestra estratificada de `N_STUDIES` estudios (semilla 42), **todas** sus cabeceras. Sin DICOM: auditoría del pipeline
(2019 estudios, 48 cortes centrales por estudio). El número real queda impreso y se propaga al informe."""))
C.append(code('''N_STUDIES = 300          # 0 = todos los estudios (más lento: ~2019 estudios × ~350 cortes)
labels = pd.read_csv(TRAIN_CSV)[["StudyInstanceUID", "patient_overall"]]
SOURCE = "dicom" if HAS_DICOM else "auditoria"
if HAS_DICOM:
    from tqdm.auto import tqdm
    uids = E.sample_studies(TRAIN_CSV, N_STUDIES)
    cache = OUT / "metadata_slices.csv.gz"
    if cache.exists() and set(pd.read_csv(cache, usecols=["StudyInstanceUID"]).StudyInstanceUID) == set(uids):
        slices = pd.read_csv(cache)
    else:
        rows = [r for u in tqdm(uids) for r in E.read_study_headers(DICOM_DIR / u)]
        slices = pd.DataFrame(rows); slices.drop(columns=["error"], errors="ignore").to_csv(cache, index=False)
    n_err = int(slices["error"].notna().sum()) if "error" in slices else 0
    studies, omit = E.studies_from_headers(slices.dropna(subset=["rows"])), []
    meta = dict(fuente="cabeceras DICOM (Kaggle)", cortes=f"{len(slices)} (ilegibles: {n_err})", muestreo="estratificada por patient_overall", limitacion="Solo cabeceras en esta etapa; píxeles en la sección 5.")
else:
    audit = pd.read_csv(AUDIT); studies, omit = E.studies_from_audit(audit)
    meta = dict(fuente=f"auditoría del pipeline ({AUDIT.name})", cortes=f"{len(audit)} (48 centrales por estudio)", muestreo="todos los estudios de la partición v2",
                limitacion="Sin DICOM no se leen cabeceras de todos los cortes (ver Dependencias, punto 5).")
studies = studies.merge(labels, on="StudyInstanceUID", how="left")
print(f"Estudios analizados: {len(studies)} | fuente: {meta['fuente']} | cortes: {meta['cortes']}")
print(studies.patient_overall.value_counts(dropna=False).rename("estudios por patient_overall").to_frame())'''))

C.append(md("## 3. Estadística descriptiva"))
C.append(code('''cols = [c for c in ["n_slices", "thickness", "pixel_mm", "spacing_z", "z_range_mm", "kvp"] if c in studies]
display(studies[cols].describe().round(3))
for c in ["rows", "cols", "intercept", "slope", "bits", "transfer_syntax"] + [c for c in ["manufacturer", "kernel"] if c in studies]:
    print(f"\\n— {c}"); print(studies[c].value_counts(dropna=False).head(8).to_string())'''))

C.append(md("## 4. Visualizaciones de metadatos y calidad"))
C.append(code('''FIG = OUT / "figures"
E.fig_distribuciones(studies, FIG); E.fig_boxplots(studies, labels, FIG); E.fig_protocolos(studies, FIG); E.fig_consistencia(studies, FIG)
for f in sorted(FIG.glob("0[1-4]_*.png")): display(Image(str(f)))'''))
C.append(code('''checks = E.quality_checks(studies)
tab = pd.DataFrame(checks).T[["descripcion", "n", "pct"]]; display(tab)
for k, v in checks.items():
    if v["n"]: print(f"{k}: ejemplos {[u.split('.')[-1] for u in v['ejemplos']]}")
studies.to_csv(OUT / "metadata_studies.csv", index=False)
(OUT / "quality_report.json").write_text(json.dumps(dict(meta=meta, n_estudios=len(studies), checks=checks, checks_omitidos=omit), indent=2, ensure_ascii=False))
E.write_report(studies, checks, omit, meta, OUT)'''))

C.append(md("""## 5. Intensidades (HU) y antes/después del preprocesamiento
Se lee el **corte central** de una submuestra: todos los estudios con `RescaleIntercept` ∉ {−1024, 0} (hasta 6) más estudios aleatorios
(semilla 42) hasta completar 24. Se usan `to_hu`, `bone_window` y `_window_resize` de `s03_preprocess.py` (solo importadas). Los errores de lectura se registran, no se ocultan."""))
C.append(code('''import pydicom, cv2
if not HAS_DICOM:
    print("SECCIÓN OMITIDA: no hay píxeles DICOM en este entorno (ver Dependencias, punto 5).")
    inten = pd.DataFrame()
else:
    from s03_preprocess import to_hu, bone_window, _window_resize
    rng = np.random.default_rng(42)
    atip = studies[~studies.intercept.isin([-1024.0, 0.0])].StudyInstanceUID.tolist()[:6]
    resto = [u for u in studies.StudyInstanceUID if u not in atip]
    pick = atip + list(rng.choice(resto, size=min(24 - len(atip), len(resto)), replace=False))
    recs, fails, cache_hu = [], [], {}
    for u in pick:
        try:
            files = sorted((DICOM_DIR / u).glob("*.dcm"), key=lambda p: int(p.stem))
            ds = pydicom.dcmread(str(files[len(files) // 2]))
            hu, slope, icpt, _ = to_hu(ds); cache_hu[u] = hu
            grp = "intercept −8192/−10240 o slope≠1" if (slope != 1 or icpt not in (-1024.0, 0.0)) else f"intercept {int(icpt)}"
            recs.append(dict(StudyInstanceUID=u, grupo=grp, slope=slope, intercept=icpt, hu_min=hu.min(), hu_p1=np.percentile(hu, 1), hu_p50=np.percentile(hu, 50),
                             hu_p99=np.percentile(hu, 99), hu_max=hu.max(), frac_aire=(hu < -900).mean(), frac_hueso=(hu > 300).mean(), raw_max=int(ds.pixel_array.max())))
        except Exception as e:
            fails.append(dict(StudyInstanceUID=u, error=str(e)[:150]))
    inten = pd.DataFrame(recs); inten.to_csv(OUT / "intensidades_corte_central.csv", index=False)
    print(f"Leídos: {len(inten)} | fallos: {len(fails)}"); display(inten.groupby("grupo")[["hu_min", "hu_p1", "hu_p50", "hu_p99", "hu_max", "frac_aire", "frac_hueso"]].median().round(3))
    if fails: display(pd.DataFrame(fails))'''))
C.append(code('''if len(inten):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for g, d in inten.groupby("grupo"):
        for u in d.StudyInstanceUID: ax[0].hist(cache_hu[u].ravel(), bins=200, range=(-1100, 2000), histtype="step", alpha=.6, label=g if u == d.StudyInstanceUID.iloc[0] else None, color=f"C{list(inten.grupo.unique()).index(g)}")
    ax[0].set(yscale="log", title="Histograma de HU (corte central) por grupo de rescale", xlabel="HU", ylabel="píxeles (log)"); ax[0].legend(fontsize=8)
    inten.boxplot(column="hu_p99", by="grupo", ax=ax[1], grid=False); ax[1].set(title="Percentil 99 de HU por grupo", xlabel="", ylabel="HU"); ax[1].tick_params(axis="x", labelsize=7)
    fig.suptitle(""); fig.tight_layout(); fig.savefig(FIG / "05_intensidades_hu.png", dpi=150, bbox_inches="tight"); plt.show()'''))
C.append(code('''if len(inten):
    lab = labels.set_index("StudyInstanceUID").patient_overall
    sel = [u for u in cache_hu if lab.get(u) == 0][:2] + [u for u in cache_hu if lab.get(u) == 1][:2]
    sel += [u for u in atip if u in cache_hu][:1]
    fig, ax = plt.subplots(len(sel), 2, figsize=(7, 3.4 * len(sel)))
    ax = np.atleast_2d(ax)
    for i, u in enumerate(sel):
        hu = cache_hu[u]; lo, hi = np.percentile(hu, [0.5, 99.5])
        ax[i, 0].imshow(np.clip(hu, lo, hi), cmap="gray"); ax[i, 0].set_title(f"Antes · HU recortado p0.5–p99.5\\n…{u.split('.')[-1]} (y={lab.get(u)})", fontsize=8)
        ax[i, 1].imshow(_window_resize(hu), cmap="gray", vmin=0, vmax=1); ax[i, 1].set_title("Después · ventana ósea L400/W1800, 224×224", fontsize=8)
        for a in ax[i]: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "06_antes_despues_preprocesamiento.png", dpi=150, bbox_inches="tight"); plt.show()
    print("Estudios mostrados:", sel, "| el último es un protocolo atípico de rescale si la lista no está vacía:", bool(atip))'''))

C.append(md("## 6. Interpretación (generada a partir de los valores calculados)"))
C.append(code('''n = len(studies); f = lambda k: checks[k]
txt = [f"**Muestra.** {n} estudios ({meta['fuente']}); cortes por estudio: mediana {studies.n_slices.median():.0f} (rango {studies.n_slices.min():.0f}–{studies.n_slices.max():.0f}).",
       f"**Adquisición.** SliceThickness mediano {studies.thickness.median():.3g} mm ({studies.thickness.nunique()} valores distintos); PixelSpacing mediano {studies.pixel_mm.median():.3g} mm (rango {studies.pixel_mm.min():.3g}–{studies.pixel_mm.max():.3g}).",
       f"**Matriz.** {f('no_512x512')['n']} estudios ({f('no_512x512')['pct']} %) no son 512×512; el redimensionado a 224×224 cambia el tamaño físico del píxel de forma distinta según PixelSpacing.",
       f"**Rescale.** {f('intercept_atipico')['n']} estudios ({f('intercept_atipico')['pct']} %) tienen RescaleIntercept distinto de −1024/0 y {f('slope_distinto_1')['n']} ({f('slope_distinto_1')['pct']} %) RescaleSlope ≠ 1: requieren verificación visual de que el HU resultante es plausible (ver sección 5).",
       f"**Consistencia interna.** Estudios con dimensiones variables: {f('shape_variable')['n']}; con rescale variable: {f('rescale_variable')['n']}; con grosor variable: {f('grosor_variable')['n']}.",
       f"**Atípicos (IQR×1.5).** n.º de cortes: {f('outlier_iqr_n_slices')['n']}; PixelSpacing: {f('outlier_iqr_pixel_mm')['n']}."]
if len(inten): txt.append(f"**Intensidades.** Corte central de {len(inten)} estudios; medianas de p99 de HU por grupo en la tabla de la sección 5.")
display(Markdown("\\n\\n".join(f"- {t}" for t in txt)))'''))

C.append(md("## 7. Limitaciones"))
C.append(code('''lim = [f"El análisis se basa en {len(studies)} estudios ({meta['muestreo']}); no es el conjunto completo salvo que `N_STUDIES = 0` o se use la auditoría v2.",
       "Las comprobaciones sobre cabeceras no equivalen a una revisión clínica de la calidad de imagen.",
       "Las intensidades se evaluaron solo en el corte central de una submuestra; no se midieron artefactos (metal, movimiento).",
       "La comparación por `patient_overall` es descriptiva: sin pruebas de hipótesis ni control de confusores de protocolo.",
       meta["limitacion"]]
if not HAS_DICOM: lim.append("Esta ejecución NO tuvo acceso a píxeles: la sección 5 no se ejecutó.")
display(Markdown("\\n".join(f"- {l}" for l in lim)))'''))

C.append(md("## 8. Exportación de artefactos"))
C.append(code('''files = sorted(p for p in OUT.rglob("*") if p.is_file() and "_scratch" not in p.parts and p.name != "MANIFEST_EDA.json")
import hashlib
man = dict(fuente=SOURCE, n_estudios=int(len(studies)), semilla=42, archivos=[dict(ruta=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(OUT / "MANIFEST_EDA.json").write_text(json.dumps(man, indent=2)); print(*[a["ruta"] for a in man["archivos"]], sep="\\n")'''))

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
out = Path(__file__).resolve().parents[1] / "02_eda_imagenes.ipynb"
nbf.write(nb, out); print("escrito", out)
