"""Obtiene estudios DICOM REALES de la partición de PRUEBA para validar la inferencia de la app.

Crea (o actualiza) un notebook PRIVADO y NUEVO en la cuenta de Kaggle del usuario
(`<usuario>/rsna-app-exportar-estudios-prueba`; no modifica el notebook de entrenamiento), que
copia los estudios elegidos de `train_images/` a un zip en su Output. Luego espera a que termine,
descarga el zip y lo descomprime en experimento_dl/data/dicom/<UID>/ (no versionado).

Requisito: credenciales de Kaggle configuradas por el usuario (KAGGLE_API_TOKEN o ~/.kaggle/kaggle.json)
y reglas de la competencia aceptadas.
    .venv/bin/python experimento_dl/kaggle/exportar_estudios_prueba.py [--n 3] [--user <usuario>]
Los UID se eligen de artifacts/v2/results/splits.csv (split == test): los de menos cortes,
equilibrando clases, para minimizar la descarga. No se usan etiquetas para nada más.
"""
import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
WORK = Path(__file__).resolve().parent / "exportar_estudios"
DEST = ROOT / "data" / "dicom"
SLUG = "rsna-app-exportar-estudios-prueba"
COMP = "rsna-2022-cervical-spine-fracture-detection"
KAGGLE = str(Path(sys.executable).parent / "kaggle")


def choose_uids(n):
    sp = pd.read_csv(REPO / "artifacts/v2/results/splits.csv")
    md = pd.read_csv(REPO / "artifacts/v2/results/dicom_metadata.csv.gz")
    nf = md.groupby("StudyInstanceUID").n_files_study.first().rename("n").reset_index()
    t = sp[sp.split == "test"].merge(nf).sort_values("n")
    neg, pos = t[t.patient_overall == 0], t[t.patient_overall == 1]
    k_pos = n // 2
    return list(neg.StudyInstanceUID[: n - k_pos]) + list(pos.StudyInstanceUID[:k_pos])


def build_notebook(uids):
    code = f'''import os, glob, shutil
UIDS = {uids!r}
cands = [p for p in glob.glob("/kaggle/input/**/train_images", recursive=True) if os.path.isdir(p)]
print("Entradas:", os.listdir("/kaggle/input"), "· train_images:", cands)
assert cands, "La competencia no está montada como entrada"
base = cands[0]
for uid in UIDS:
    src = os.path.join(base, uid)
    print(uid, len(os.listdir(src)), "archivos")
    shutil.copytree(src, f"/kaggle/working/estudios/{{uid}}", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/estudios_prueba", "zip", "/kaggle/working/estudios")
shutil.rmtree("/kaggle/working/estudios")
print("zip:", round(os.path.getsize("/kaggle/working/estudios_prueba.zip") / 1e6, 1), "MB")'''
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# Exportar estudios de prueba (RSNA 2022) para validar la app\n\n"
                                 "Notebook auxiliar de P1 (app). Solo copia DICOM de la partición de **prueba** "
                                 "(`artifacts/v2/results/splits.csv`) a un zip; no entrena ni evalúa."),
        nbf.v4.new_code_cell(code)],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    WORK.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, WORK / "exportar_estudios_prueba.ipynb")


def kaggle(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"kaggle {' '.join(args)} falló:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def username(given=None):
    import os
    if given or os.environ.get("KAGGLE_USERNAME"):
        return given or os.environ["KAGGLE_USERNAME"]
    out = kaggle("config", "view", check=False)
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("No se encontró el usuario de Kaggle: usa --user <usuario> (el de kaggle.com/<usuario>)\n" + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--user", default=None, help="usuario de Kaggle (si config view no lo muestra)")
    a = ap.parse_args()
    uids = choose_uids(a.n)
    print("Estudios de prueba elegidos:", uids)
    build_notebook(uids)
    user = username(a.user)
    meta = dict(id=f"{user}/{SLUG}", title=SLUG, code_file="exportar_estudios_prueba.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_internet=False,
                dataset_sources=[], competition_sources=[COMP], kernel_sources=[])
    (WORK / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    out_push = kaggle("kernels", "push", "-p", str(WORK))
    print(out_push)
    if "error" in out_push.lower():  # el CLI devuelve código 0 aunque el push falle
        sys.exit("No se pudo subir el notebook (¿reglas de la competencia aceptadas?).")
    t0 = time.time()
    while time.time() - t0 < 1800:
        st = kaggle("kernels", "status", meta["id"], check=False)
        print(f"[{time.time() - t0:5.0f}s] {st.strip()}")
        if "complete" in st.lower():
            break
        if "error" in st.lower() or "cancel" in st.lower():
            sys.exit("El notebook falló en Kaggle; revisa su log en la web.")
        time.sleep(20)
    else:
        sys.exit("Tiempo de espera agotado (30 min); revisa el notebook en kaggle.com.")
    out = WORK / "output"
    out.mkdir(exist_ok=True)
    print(kaggle("kernels", "output", meta["id"], "-p", str(out), "--force"))
    z = out / "estudios_prueba.zip"
    if not z.exists():
        sys.exit(f"No se descargó {z}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(DEST)
    for u in uids:
        print(u, len(list((DEST / u).glob("*.dcm"))), "DICOM en", (DEST / u).relative_to(REPO))


if __name__ == "__main__":
    main()
