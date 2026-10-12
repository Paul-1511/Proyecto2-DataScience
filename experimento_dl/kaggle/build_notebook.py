"""Genera rsna_entrenamiento_v2.ipynb (notebook para Kaggle). Ejecutar: python build_notebook.py"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# RSNA 2022 · Entrenamiento y evaluación v2 (ResNet18 + MIL vs ResNet18 + BiGRU)

Notebook reproducible del repositorio `Paul-1511/Proyecto2-DataScience` (rama `feature/entrenamiento`,
carpeta `experimento_dl/`). Ejecuta el **mismo código versionado** (no hay lógica duplicada aquí):

| Paso | Script | Salida en `/kaggle/working` |
|---|---|---|
| Preprocesamiento DICOM 2.5D | `s03_preprocess.py` | `cache_v2/` (~5-6 GB), `results_v2/dicom_metadata.csv`, fallos |
| Modelo A: ResNet18 + MIL | `s04_train_eval.py --model mil` | `weights/mil_final.pt`, métricas, figuras |
| Modelo B: ResNet18 + BiGRU | `s04_train_eval.py --model bigru` | `weights/bigru_final.pt`, métricas, figuras |
| Comparación | `s05_compare_report.py` | `results_v2/comparacion_modelos.csv`, figuras |
| Control de calidad | `s07_qc.py` | `results_v2/qc_report.json` |

**Configuración necesaria del notebook (panel derecho → Settings):**
1. *Add Input* → competición **RSNA 2022 Cervical Spine Fracture Detection** (aceptar reglas antes).
2. *Accelerator* → **GPU T4 x2** (o T4). Evitar P100 si la versión de PyTorch no incluye `sm_60` (se verifica abajo).
3. *Internet* → **On** (clonar el repo, `pip install` de decodificadores JPEG y pesos ImageNet de ResNet18).
   Requiere cuenta verificada por teléfono. Sin internet: ver la celda de pesos (alternativa offline).
4. Para ejecuciones largas: **Save Version → Save & Run All (Commit)**; las salidas quedan en *Output*.

Particiones: `results_v2/splits.csv` versionado en el repo (2019 estudios, 64/16/20 = 1292/323/404,
estratificado por `patient_overall`, semilla 42). Este notebook **no** las regenera ni las modifica.
La prueba se evalúa una sola vez, con época y umbral elegidos en validación.
""")

code("""
# ===== Parámetros =====
REPO_URL = "https://github.com/Paul-1511/Proyecto2-DataScience.git"
BRANCH = "feature/entrenamiento"
COMP = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
WORK = "/kaggle/working"
EPOCHS, PATIENCE = 15, 4          # máximo de épocas y early stopping (AUC de validación)
BS, ACCUM = 4, 2                  # estudios por paso y acumulación (batch efectivo 8 estudios)
MODELS = ["mil", "bigru"]         # Modelo A primero (línea base), luego Modelo B
CACHE_INPUT = None                # ruta a un cache_v2 de una versión anterior (Output) para no repreprocesar
KEEP_CACHE_IN_OUTPUT = True       # conservar cache_v2 en Output (~5-6 GB) para reutilizarlo
""")

code("""
import os, sys, subprocess, shutil, time, json, glob
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i),
              f"{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    assert arch in torch.cuda.get_arch_list() or cap >= (7, 0), (
        f"La GPU ({arch}) no está soportada por este PyTorch {torch.cuda.get_arch_list()}: elegir GPU T4.")
    x = torch.randn(256, 256, device="cuda"); print("prueba GPU ok:", float((x @ x).sum()) == float((x @ x).sum()))
else:
    print("AVISO: sin GPU. Activar Accelerator = GPU T4 antes de entrenar.")
print("CPUs:", os.cpu_count())
for p in [WORK, "/tmp"]:
    t, u, f = shutil.disk_usage(p); print(f"disco {p}: libre {f/1e9:.1f} GB de {t/1e9:.1f} GB")
""")

code("""
# ===== Datos de la competición montados en /kaggle/input =====
import pandas as pd
assert os.path.isdir(COMP), "Falta el input de la competición (Add Input → RSNA 2022 Cervical Spine Fracture Detection)"
print(sorted(os.listdir(COMP)))
train_k = pd.read_csv(f"{COMP}/train.csv")
studies = os.listdir(f"{COMP}/train_images")
print("train.csv:", train_k.shape, "| carpetas en train_images:", len(studies))
print("estudios de train.csv sin carpeta:", len(set(train_k.StudyInstanceUID) - set(studies)))
""")

code("""
# ===== Dependencias: decodificadores JPEG Lossless para pydicom =====
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom>=3", "pylibjpeg", "pylibjpeg-libjpeg",
                "python-gdcm"], check=False)
import pydicom
print("pydicom", pydicom.__version__)
# Verificación: decodificar un corte de varios estudios y registrar sus Transfer Syntax
seen = {}
for uid in sorted(studies)[:40]:
    f = sorted(glob.glob(f"{COMP}/train_images/{uid}/*.dcm"), key=lambda p: int(os.path.basename(p)[:-4]))[0]
    ds = pydicom.dcmread(f)
    ts = ds.file_meta.TransferSyntaxUID.name
    if ts not in seen:
        seen[ts] = ds.pixel_array.shape
print("Transfer syntaxes decodificadas:", seen)
""")

code("""
# ===== Código del repositorio (rama versionada) =====
os.chdir(WORK)
if not os.path.isdir("repo"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO_URL, "repo"], check=True)
CODE = f"{WORK}/repo/experimento_dl"
print(subprocess.run(["git", "-C", "repo", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
# train.csv del repo == train.csv de Kaggle (mismas etiquetas que las particiones)
repo_train = pd.read_csv(f"{WORK}/repo/train.csv")
assert repo_train.sort_values("StudyInstanceUID").reset_index(drop=True).equals(
    train_k.sort_values("StudyInstanceUID").reset_index(drop=True)), "train.csv difiere del de Kaggle"
splits = pd.read_csv(f"{CODE}/results_v2/splits.csv")
print(splits.groupby("split").patient_overall.agg(["size", "sum"]))
""")

code("""
# ===== Entorno del pipeline (rutas a /kaggle/input y /kaggle/working) =====
env = dict(os.environ,
           RSNA_EXP="v2",
           RSNA_DICOM_DIR=f"{COMP}/train_images",
           RSNA_TRAIN_CSV=f"{COMP}/train.csv",
           RSNA_CACHE_DIR=CACHE_INPUT or f"{WORK}/cache_v2",
           RSNA_RESULTS_DIR=f"{WORK}/results_v2",
           RSNA_WEIGHTS_DIR=f"{WORK}/weights",
           PYTHONUNBUFFERED="1")
os.makedirs(env["RSNA_RESULTS_DIR"], exist_ok=True)
shutil.copy(f"{CODE}/results_v2/splits.csv", f"{env['RSNA_RESULTS_DIR']}/splits.csv")
shutil.copy(f"{CODE}/results_v2/splits_summary.csv", f"{env['RSNA_RESULTS_DIR']}/splits_summary.csv")

# Pesos ImageNet de ResNet18: torchvision (internet) o archivo local en /kaggle/input (offline)
try:
    import torchvision
    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    print("Pesos ImageNet descargados por torchvision")
except Exception as e:
    cands = glob.glob("/kaggle/input/**/resnet18*.pth", recursive=True)
    assert cands, f"Sin internet y sin resnet18*.pth en /kaggle/input: {e}"
    env["RESNET18_WEIGHTS"] = cands[0]; print("Pesos ImageNet locales:", cands[0])

def run(cmd):
    t = time.time()
    p = subprocess.run([sys.executable, *cmd], cwd=CODE, env=env)
    print(f"--> {' '.join(cmd)}: código {p.returncode}, {(time.time()-t)/60:.1f} min")
    assert p.returncode == 0
""")

code("""
# ===== Supuesto verificado: nombre de archivo == InstanceNumber (regla de muestreo por instancia) =====
bad = 0
for uid in splits.StudyInstanceUID.sample(30, random_state=0):
    for f in glob.glob(f"{COMP}/train_images/{uid}/*.dcm")[:5]:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        bad += int(os.path.basename(f)[:-4]) != int(ds.InstanceNumber)
print("archivos con nombre != InstanceNumber:", bad)
""")

code("""
# ===== Preprocesamiento DICOM 2.5D (s03) — incremental; los fallos se registran por estudio =====
if CACHE_INPUT is None:
    run(["s03_preprocess.py", "--workers", str(os.cpu_count())])
fails = pd.read_csv(f"{env['RSNA_RESULTS_DIR']}/preprocess_failures.csv")
print("estudios con error:", len(fails)); display(fails.head(20))
print(pd.read_csv(f"{env['RSNA_RESULTS_DIR']}/orden_z_auditoria.csv").z_direction_vs_instance.value_counts())
""")

code("""
# ===== Modelo A (ResNet18 + MIL) y Modelo B (ResNet18 + BiGRU): mismas particiones y configuración =====
for m in MODELS:
    run(["s04_train_eval.py", "--model", m, "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
         "--bs", str(BS), "--accum", str(ACCUM), "--workers", str(min(4, os.cpu_count()))])
    print(json.dumps(json.load(open(f"{env['RSNA_RESULTS_DIR']}/metrics_{m}.json"))["test_thr_val_youden"], indent=1))
""")

code("""
# ===== Comparación y control de calidad =====
run(["s05_compare_report.py"])
display(pd.read_csv(f"{env['RSNA_RESULTS_DIR']}/comparacion_modelos.csv"))
run(["s07_qc.py", "--n-infer", "5"])
""")

code("""
# ===== Empaquetado de artefactos en /kaggle/working (pestaña Output) =====
from IPython.display import Image
for f in sorted(glob.glob(f"{env['RSNA_RESULTS_DIR']}/figures/comparacion_*.png")):
    display(Image(f))
for f in glob.glob(f"{WORK}/weights/*_best_state.pt"):   # estados intermedios: el final ya los contiene
    os.remove(f)
shutil.make_archive(f"{WORK}/results_v2", "zip", f"{WORK}/results_v2")
if not KEEP_CACHE_IN_OUTPUT and CACHE_INPUT is None:
    shutil.rmtree(f"{WORK}/cache_v2")
shutil.rmtree(f"{WORK}/repo")   # el código ya está versionado en GitHub
for p in sorted(glob.glob(f"{WORK}/*")):
    s = sum(os.path.getsize(x) for x in glob.glob(p + "/**", recursive=True) if os.path.isfile(x)) if os.path.isdir(p) else os.path.getsize(p)
    print(f"{p}: {s/1e6:.1f} MB")
""")

md("""
## Recuperar los artefactos
- **Output** de la versión guardada: `weights/mil_final.pt`, `weights/bigru_final.pt` (~45 MB c/u),
  `results_v2.zip` (métricas JSON/CSV, predicciones, historiales, figuras, `qc_report.json`).
- En el repositorio se versiona `results_v2/` (sin pesos ni caché). Los checkpoints se comparten como
  Output de Kaggle (o artefacto privado), no en GitHub.
- Inferencia: `inference.py` (`load_classifier` + `predict_study`), ver `INTEGRACION_STREAMLIT.md`.
""")

nb = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python"),
    kaggle=dict(accelerator="gpu", isInternetEnabled=True, dataSources=[], isGpuEnabled=True)),
    nbformat=4, nbformat_minor=5)
out = Path(__file__).with_name("rsna_entrenamiento_v2.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("escrito", out)
