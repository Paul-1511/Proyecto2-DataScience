"""Valida la inferencia de la app con los checkpoints reales y deja la evidencia en docs/app/.

    .venv/bin/python experimento_dl/tools/validar_inferencia.py [--max 10]

Requisitos: artifacts/v2/weights/{mil,bigru}_final.pt (SHA-256 de MANIFEST.json) y estudios RSNA en
experimento_dl/data/dicom/<StudyInstanceUID>/*.dcm. Salidas (solo si hay checkpoints reales):
  docs/app/validacion_inferencia.csv   una fila por estudio × modelo
  docs/app/validacion_inferencia.json  resumen: reproducibilidad frente a la evaluación, tiempos, acuerdo
Sin checkpoints imprime PENDIENTE y termina con código 2 (no genera resultados).
"""
import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app_core as core  # noqa: E402

OUT = core.REPO / "docs" / "app"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10, help="máximo de estudios locales a evaluar")
    ap.add_argument("--out", default=str(OUT), help="carpeta de salida")
    a = ap.parse_args(argv)
    out = Path(a.out)
    status = {m: core.checkpoint_status(m) for m in core.MODELS}
    for m, s in status.items():
        print(f"{core.NAMES[m]}: checkpoint {s['estado']}")
    if any(s["estado"] != "DISPONIBLE" for s in status.values()):
        print("PENDIENTE: copiar los checkpoints publicados en artifacts/v2/weights/. No se genera evidencia.")
        sys.exit(2)
    studies = core.local_studies()[: a.max]
    if not studies:
        print("PENDIENTE: no hay estudios en experimento_dl/data/dicom/<UID>/.")
        sys.exit(2)
    loaded, skipped = core.load_models(core.MODELS)
    if skipped:
        sys.exit(f"Error al cargar: {skipped}")
    df = core.predict_local_batch(studies, loaded, progress=lambda f, u: print(f"  {f:4.0%} {u}"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "validacion_inferencia.csv", index=False)
    ok = df.dropna(subset=["model"]) if "model" in df else df.iloc[0:0]
    chk = ok.dropna(subset=["reproduce_evaluacion"]) if "reproduce_evaluacion" in ok else ok.iloc[0:0]
    piv = ok.pivot_table(index="StudyInstanceUID", columns="model", values="prediccion", aggfunc="first")
    summary = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        device=str(next(iter(loaded.values()))["device"]), platform=platform.platform(),
        checkpoints={m: s["sha256"] for m, s in status.items()},
        n_estudios=len(studies), n_errores=int(df.error.notna().sum()) if "error" in df else 0,
        reproducibilidad=dict(n=len(chk), reproduce=int(chk.reproduce_evaluacion.sum()) if len(chk) else 0,
                              max_delta=float(chk.delta_evaluacion.max()) if len(chk) else None,
                              tolerancia=core.REPRO_TOL),
        segundos_por_estudio={m: float(ok[ok.model == m].segundos.mean()) for m in core.MODELS},
        acuerdo_A_B=float((piv["mil"] == piv["bigru"]).mean()) if {"mil", "bigru"} <= set(piv) else None,
        nota="Validación técnica de la app (reproducción de la evaluación); no es una nueva evaluación del modelo.")
    (out / "validacion_inferencia.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.exit(0 if not len(chk) or summary["reproducibilidad"]["reproduce"] == len(chk) else 1)


if __name__ == "__main__":
    main()
