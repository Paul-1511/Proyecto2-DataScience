"""Empaqueta el código versionado (HEAD) para subirlo como Kaggle Dataset.

Usa `git archive` sobre el commit actual: solo entra código ya commiteado (sin datos, pesos ni
cachés). Resultado (no versionado): experimento_dl/data/kaggle_bundle/rsna_codigo.zip, que contiene
rsna_codigo/{train.csv, VERSION.txt, experimento_dl/...}. En el notebook:
CODE_DATASET = "/kaggle/input/<dataset>/rsna_codigo".
Uso: python kaggle/make_code_bundle.py   (desde experimento_dl/, con el árbol limpio)
"""
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experimento_dl" / "data" / "kaggle_bundle"
PATHS = ["train.csv", "experimento_dl/common.py", "experimento_dl/inference.py",
         "experimento_dl/requirements.txt", "experimento_dl/results_v2/splits.csv",
         "experimento_dl/results_v2/splits_summary.csv"] + \
        [f"experimento_dl/s0{i}" for i in (1, 3, 4, 5, 7)]


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout


def main():
    if git("status", "--porcelain", "--", "experimento_dl", "train.csv").strip():
        raise SystemExit("Hay cambios sin commitear en experimento_dl/: commitear antes de empaquetar.")
    head = git("log", "-1", "--format=%H %s").strip()
    files = [p for p in git("ls-files", *[p if "." in Path(p).name else p + "*" for p in PATHS]).split()]
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD", *files],
                         capture_output=True, check=True).stdout
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "rsna_codigo.zip"
    with tarfile.open(fileobj=io.BytesIO(tar)) as t, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for m in t.getmembers():
            if m.isfile():
                z.writestr(f"rsna_codigo/{m.name}", t.extractfile(m).read())
        z.writestr("rsna_codigo/VERSION.txt", f"commit {head}\n")
    print(f"{dst} ({dst.stat().st_size / 1e3:.0f} KB) desde {head}")
    for n in sorted(zipfile.ZipFile(dst).namelist()):
        print("  ", n)


if __name__ == "__main__":
    main()
