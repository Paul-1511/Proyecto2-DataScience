#!/usr/bin/env python3
"""Genera informe/generated/eda_tablas.tex y eda_cifras.tex a partir de los artefactos del EDA de imágenes.
Ninguna cifra se escribe a mano: todo se lee de metadata_studies.csv, quality_report.json y auditoria_pipeline.json.
Uso: python informe/tools/gen_tablas_eda.py [carpeta_eda]   (por defecto artifacts/eda_imagenes/auditoria_v2;
tras ejecutar el notebook 02 en Kaggle, pasar artifacts/eda_imagenes)."""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "eda_imagenes" / "auditoria_v2"
OUT = ROOT / "informe" / "generated"; OUT.mkdir(exist_ok=True)
s = pd.read_csv(SRC / "metadata_studies.csv")
q = json.loads((SRC / "quality_report.json").read_text(encoding="utf-8"))
a = json.loads((SRC / "auditoria_pipeline.json").read_text(encoding="utf-8")) if (SRC / "auditoria_pipeline.json").exists() else {}

LAB = {  # etiquetas LaTeX-seguras (las descripciones originales usan símbolos Unicode)
    "no_512x512": r"Matriz distinta de $512\times512$",
    "intercept_atipico": r"RescaleIntercept $\notin\{-1024,0\}$ (p.\,ej. $-8192$/$-10240$)",
    "slope_distinto_1": r"RescaleSlope $\neq 1$",
    "shape_variable": "Dimensiones de matriz variables dentro del estudio",
    "rescale_variable": "Slope/intercept variables dentro del estudio",
    "grosor_variable": "SliceThickness variable dentro del estudio",
    "not_16bit_signed_or_unsigned": r"BitsStored fuera de $\{12,16\}$",
    "monochrome1": r"PhotometricInterpretation $\neq$ MONOCHROME2",
    "outlier_iqr_n_slices": r"At\'ipico (IQR$\times$1.5) en n.$^\circ$ de cortes",
    "outlier_iqr_thickness": r"At\'ipico (IQR$\times$1.5) en SliceThickness",
    "outlier_iqr_pixel_mm": r"At\'ipico (IQR$\times$1.5) en PixelSpacing",
    "outlier_iqr_spacing_z": r"At\'ipico (IQR$\times$1.5) en espaciado $z$",
}
rows = [f"{LAB[k]} & {v['n']} & {v['pct']:.2f} \\\\" for k, v in q["checks"].items() if k in LAB]
tab_checks = ("\\begin{tabular}{lrr}\n\\toprule\nComprobaci\\'on & Estudios & \\% \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

cols = [("n_slices", "Cortes por estudio"), ("thickness", "SliceThickness (mm)"), ("pixel_mm", "PixelSpacing (mm)"), ("spacing_z", r"Espaciado $z$ medido (mm)")]
rows = []
for c, n in cols:
    d = s[c].dropna(); rows.append(f"{n} & {d.mean():.3f} & {d.std():.3f} & {d.min():.3f} & {d.median():.3f} & {d.max():.3f} \\\\")
tab_desc = ("\\begin{tabular}{lrrrrr}\n\\toprule\nVariable & Media & D.\\,E. & M\\'in. & Mediana & M\\'ax. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

(OUT / "eda_tablas.tex").write_text("% GENERADO por informe/tools/gen_tablas_eda.py: no editar a mano\n"
    "\\newcommand{\\tablaChecks}{" + tab_checks + "}\n\\newcommand{\\tablaDesc}{" + tab_desc + "}\n", encoding="utf-8")

def m(x): return str(x).replace("_", r"\_")
dz = a.get("z_direccion", {}); ct = a.get("z_direccion_vs_fractura", {})
pos_c = ct.get("1", {}).get("creciente", 0); neg_c = ct.get("0", {}).get("creciente", 0)
pos_t = sum(ct.get("1", {}).values()); neg_t = sum(ct.get("0", {}).values())
C = {
 "NEstudios": q["n_estudios"], "Fuente": q["meta"]["fuente"].replace("_", r"\_"), "CortesLeidos": q["meta"]["cortes"].split(" ")[0],
 "Neg": int((s.patient_overall == 0).sum()), "Pos": int((s.patient_overall == 1).sum()),
 "NCortesMed": f"{s.n_slices.median():.0f}", "NCortesMin": f"{s.n_slices.min():.0f}", "NCortesMax": f"{s.n_slices.max():.0f}",
 "ZDecr": dz.get("decreciente", "?"), "ZCrec": dz.get("creciente", "?"), "ZMixto": dz.get("mixto", "?"),
 "ZNoMono": a.get("z_no_monotono", "?"), "ZAnomalos": ", ".join("\\ldots" + u for u in a.get("z_anomalos", [])),
 "PosCrecPct": f"{100*pos_c/pos_t:.1f}" if pos_t else "?", "NegCrecPct": f"{100*neg_c/neg_t:.1f}" if neg_t else "?",
 "ChiP": f"{a['z_direccion_chi2_p']:.1e}" if "z_direccion_chi2_p" in a else "?",
 "FallosPrep": a.get("fallos_preprocesamiento", "?"),
 "TMedia": a.get("tiempo_s", {}).get("mean", "?"), "TMediana": a.get("tiempo_s", {}).get("50%", "?"), "TMax": a.get("tiempo_s", {}).get("max", "?"),
}
for macro, key in {"NNoQuinientos": "no_512x512", "NInterceptAtip": "intercept_atipico", "NSlopeNoUno": "slope_distinto_1",
                   "NOutCortes": "outlier_iqr_n_slices", "NOutPixel": "outlier_iqr_pixel_mm", "NOutZ": "outlier_iqr_spacing_z",
                   "NGrosorSpacing": "grosor_vs_spacing_z"}.items():
    C[macro] = q["checks"][key]["n"]; C[macro + "Pct"] = f"{q['checks'][key]['pct']:.2f}"
(OUT / "eda_cifras.tex").write_text("% GENERADO: no editar a mano\n" + "".join(f"\\newcommand{{\\cif{k}}}{{{v}}}\n" for k, v in C.items()), encoding="utf-8")
print("OK ->", OUT, "| fuente:", SRC)
