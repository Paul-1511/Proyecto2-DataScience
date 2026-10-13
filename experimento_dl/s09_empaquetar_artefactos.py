"""Empaqueta los artefactos compartidos del experimento v2 en <repo>/artifacts/v2/ (sin reentrenar).

- config/model_<m>.json      : hiperparámetros, umbral, época elegida, AUC de validación (del checkpoint)
- config/preprocessing.json  : configuración de preprocesamiento (idéntica en ambos checkpoints)
- MANIFEST.json              : todos los archivos con tamaño y SHA-256 (incluye los checkpoints,
                               que NO se versionan; sirve para verificar la copia recibida)
Requiere los checkpoints en artifacts/v2/weights/<m>_final.pt (ignorados por git).
Uso: python s09_empaquetar_artefactos.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

ART = Path(__file__).resolve().parents[1] / "artifacts" / "v2"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    (ART / "config").mkdir(parents=True, exist_ok=True)
    pre = None
    for m in ("mil", "bigru"):
        with torch.serialization.safe_globals([TorchVersion]):
            ck = torch.load(ART / "weights" / f"{m}_final.pt", map_location="cpu", weights_only=True)
        if pre is not None and ck["preprocessing"] != pre:
            raise SystemExit("Los checkpoints no comparten preprocesamiento")
        pre = ck["preprocessing"]
        cfg = dict(model_name=ck["model_name"], threshold_val_youden=ck["threshold"], best_epoch=ck["best_epoch"],
                   val_auc=ck["val_auc"], hparams=ck["hparams"], torch_version=str(ck["torch_version"]),
                   n_params=sum(v.numel() for k, v in ck["state_dict"].items() if "num_batches_tracked" not in k
                                and not k.endswith(("running_mean", "running_var"))),
                   checkpoint=f"weights/{m}_final.pt", note=ck["note"])
        (ART / "config" / f"model_{m}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    (ART / "config" / "preprocessing.json").write_text(json.dumps(pre, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    files = sorted(p for p in ART.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source="Kaggle Notebook rsna_entrenamiento_v2 (GPU Tesla T4), código feature/entrenamiento @ 772a276",
        versioned_in_git="todo excepto weights/*.pt",
        files=[dict(path=p.relative_to(ART).as_posix(), bytes=p.stat().st_size, sha256=sha256(p),
                    in_git=p.suffix != ".pt") for p in files])
    (ART / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"{len(files)} archivos; {sum(f['bytes'] for f in manifest['files'] if f['in_git']) / 1e6:.1f} MB versionables; "
          f"{sum(f['bytes'] for f in manifest['files'] if not f['in_git']) / 1e6:.1f} MB de pesos (no versionados)")


if __name__ == "__main__":
    main()
