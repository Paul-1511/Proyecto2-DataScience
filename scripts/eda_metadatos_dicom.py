#!/usr/bin/env python3
"""Análisis independiente de metadatos DICOM — RSNA 2022 (P2: EDA de imágenes).

Dos modos de entrada (el resto del análisis es idéntico):
  --dicom-dir RUTA   Lee SOLO las cabeceras (stop_before_pixels) de TODOS los cortes de una muestra
                     reproducible de estudios (semilla 42, estratificada por patient_overall).
                     En Kaggle: RUTA=/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images
  --from-audit CSV   Reutiliza la auditoría ya generada por el pipeline de entrenamiento
                     (artifacts/v2/results/dicom_metadata.csv.gz: 48 cortes centrales por estudio).
                     Solo lectura; sirve para analizar sin acceso a los DICOM.

No lee píxeles, no modifica s03_preprocess.py ni artifacts/v2/. Salidas en --out (por defecto
artifacts/eda_imagenes/): metadata_studies.csv, quality_report.json, INFORME_CALIDAD_DICOM.md, figures/*.png.
Nada se inventa: si una columna no existe en la fuente, la comprobación correspondiente se omite y
queda registrada en quality_report.json -> "checks_omitidos".
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEED = 42
TAGS = ["InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", "SliceThickness",
        "PixelSpacing", "Rows", "Columns", "BitsStored", "PixelRepresentation", "RescaleSlope",
        "RescaleIntercept", "PhotometricInterpretation", "Modality", "Manufacturer",
        "ManufacturerModelName", "KVP", "ConvolutionKernel", "PixelPaddingValue"]


# --------------------------------------------------------------------------- muestreo y lectura
def sample_studies(train_csv: Path, n: int, seed: int = SEED) -> list[str]:
    """Muestra estratificada por patient_overall, reproducible. n<=0 -> todos los estudios."""
    tr = pd.read_csv(train_csv)[["StudyInstanceUID", "patient_overall"]].sort_values("StudyInstanceUID")
    if n <= 0 or n >= len(tr):
        return tr.StudyInstanceUID.tolist()
    rng = np.random.default_rng(seed)
    out = []
    for _, g in tr.groupby("patient_overall"):
        k = max(1, round(n * len(g) / len(tr)))
        out += rng.choice(g.StudyInstanceUID.to_numpy(), size=min(k, len(g)), replace=False).tolist()
    return sorted(out)


def read_study_headers(study_dir: Path) -> list[dict]:
    import pydicom
    rows = []
    for f in study_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=TAGS)
        except Exception as e:  # archivo ilegible: se registra, no se oculta
            rows.append(dict(StudyInstanceUID=study_dir.name, file=f.name, error=str(e)[:120]))
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        ps = getattr(ds, "PixelSpacing", None)
        rows.append(dict(
            StudyInstanceUID=study_dir.name, file=f.name,
            instance=getattr(ds, "InstanceNumber", None),
            z=float(ipp[2]) if ipp is not None and len(ipp) == 3 else np.nan,
            thickness=float(ds.SliceThickness) if getattr(ds, "SliceThickness", None) is not None else np.nan,
            ps_row=float(ps[0]) if ps is not None else np.nan, ps_col=float(ps[1]) if ps is not None else np.nan,
            rows=int(ds.Rows), cols=int(ds.Columns), bits=int(getattr(ds, "BitsStored", -1)),
            pixrep=int(getattr(ds, "PixelRepresentation", -1)),
            slope=float(getattr(ds, "RescaleSlope", np.nan)), intercept=float(getattr(ds, "RescaleIntercept", np.nan)),
            photometric=str(getattr(ds, "PhotometricInterpretation", "")), modality=str(getattr(ds, "Modality", "")),
            manufacturer=str(getattr(ds, "Manufacturer", "")), model=str(getattr(ds, "ManufacturerModelName", "")),
            kvp=float(ds.KVP) if getattr(ds, "KVP", None) not in (None, "") else np.nan,
            kernel=str(getattr(ds, "ConvolutionKernel", "")),
            axial=bool(iop is not None and abs(float(iop[0]) * float(iop[4]) - float(iop[1]) * float(iop[3])) > 0.9)
            if iop is not None else None,
            transfer_syntax=str(ds.file_meta.TransferSyntaxUID.name)))
    return rows


def studies_from_headers(slices: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for uid, g in slices.dropna(subset=["rows"]).groupby("StudyInstanceUID"):
        zs = np.sort(g.z.dropna().unique())
        dz = np.diff(zs)
        med = float(np.median(dz)) if len(dz) else np.nan
        recs.append(dict(
            StudyInstanceUID=uid, n_slices=len(g), thickness=g.thickness.median(),
            thickness_nunique=g.thickness.nunique(), pixel_mm=g.ps_row.median(),
            pixel_aniso=bool((g.ps_row - g.ps_col).abs().max() > 1e-6), rows=int(g.rows.mode()[0]),
            cols=int(g.cols.mode()[0]), shape_nunique=g.groupby(["rows", "cols"]).ngroups,
            slope=g.slope.mode()[0], intercept=g.intercept.mode()[0],
            rescale_nunique=g.groupby(["slope", "intercept"]).ngroups, spacing_z=med,
            z_gap_ratio=float(dz.max() / med) if len(dz) and med > 0 else np.nan,
            z_dup=int(len(g.z.dropna()) - len(zs)), z_range_mm=float(zs[-1] - zs[0]) if len(zs) else np.nan,
            transfer_syntax=g.transfer_syntax.mode()[0], manufacturer=g.manufacturer.mode()[0],
            kvp=g.kvp.median(), kernel=g.kernel.mode()[0], bits=int(g.bits.mode()[0]),
            non_axial=bool((g.axial == False).any()), photometric=g.photometric.mode()[0]))  # noqa: E712
    return pd.DataFrame(recs)


def studies_from_audit(audit: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Auditoría del pipeline (48 cortes centrales/estudio). n_slices sale de n_files_study."""
    a = audit.copy()
    a["ps"] = a.pixel_spacing.map(lambda s: ast.literal_eval(s)[0] if isinstance(s, str) else np.nan)
    g = a.groupby("StudyInstanceUID")
    s = pd.DataFrame(dict(
        n_slices=g.n_files_study.first(), thickness=g.slice_thickness.median(),
        thickness_nunique=g.slice_thickness.nunique(), pixel_mm=g.ps.median(),
        rows=g.rows.agg(lambda x: x.mode()[0]), cols=g.cols.agg(lambda x: x.mode()[0]),
        shape_nunique=g.apply(lambda d: d.groupby(["rows", "cols"]).ngroups, include_groups=False),
        slope=g.slope.agg(lambda x: x.mode()[0]), intercept=g.intercept.agg(lambda x: x.mode()[0]),
        rescale_nunique=g.apply(lambda d: d.groupby(["slope", "intercept"]).ngroups, include_groups=False),
        spacing_z=g.spacing_z_mm.median(), transfer_syntax=g.transfer_syntax.first(),
        bits=g.bits.agg(lambda x: x.mode()[0]), photometric=g.photometric.first())).reset_index()
    omit = ["pixel_aniso", "z_gap_ratio", "z_dup", "z_range_mm", "manufacturer", "kvp", "kernel", "non_axial"]
    return s, omit


# --------------------------------------------------------------------------- calidad
def iqr_outliers(x: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = x.quantile([.25, .75])
    return (x < q1 - k * (q3 - q1)) | (x > q3 + k * (q3 - q1))


def quality_checks(s: pd.DataFrame) -> dict:
    def chk(mask, desc):
        return dict(descripcion=desc, n=int(mask.sum()), pct=round(100 * float(mask.mean()), 2),
                    ejemplos=s.loc[mask, "StudyInstanceUID"].head(3).tolist())
    c = {
        "no_512x512": chk((s.rows != 512) | (s.cols != 512), "Matriz distinta de 512×512"),
        "intercept_atipico": chk(~s.intercept.isin([-1024.0, 0.0]),
                                 "RescaleIntercept distinto de −1024 y 0 (p. ej. −8192/−10240)"),
        "slope_distinto_1": chk(s.slope != 1.0, "RescaleSlope distinto de 1"),
        "shape_variable": chk(s.shape_nunique > 1, "Dimensiones de matriz distintas dentro del estudio"),
        "rescale_variable": chk(s.rescale_nunique > 1, "Slope/intercept distintos dentro del estudio"),
        "grosor_variable": chk(s.thickness_nunique > 1, "SliceThickness variable dentro del estudio"),
        "grosor_vs_spacing_z": chk((s.thickness - s.spacing_z).abs() > 0.05,
                                   "|SliceThickness − espaciado z medido| > 0.05 mm"),
        "not_16bit_signed_or_unsigned": chk(~s.bits.isin([12, 16]), "BitsStored fuera de {12,16}"),
        "monochrome1": chk(s.photometric != "MONOCHROME2", "PhotometricInterpretation ≠ MONOCHROME2"),
    }
    for col, lab in [("n_slices", "n.º de cortes"), ("thickness", "SliceThickness"),
                     ("pixel_mm", "PixelSpacing"), ("spacing_z", "espaciado z")]:
        if col in s:
            c[f"outlier_iqr_{col}"] = chk(iqr_outliers(s[col].dropna()).reindex(s.index, fill_value=False),
                                          f"Atípico (IQR×1.5) en {lab}")
    if "z_gap_ratio" in s:
        c["hueco_z"] = chk(s.z_gap_ratio > 1.5, "Hueco en z > 1.5× el espaciado mediano")
        c["z_duplicado"] = chk(s.z_dup > 0, "Posiciones z duplicadas")
        c["no_axial"] = chk(s.non_axial.fillna(False), "Orientación no axial")
    return c


# --------------------------------------------------------------------------- figuras
def fig_distribuciones(s, out):
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (col, t, bins) in zip(ax, [("thickness", "SliceThickness (mm)", 20), ("pixel_mm", "PixelSpacing (mm)", 25),
                                      ("n_slices", "Cortes por estudio", 30), ("spacing_z", "Espaciado z medido (mm)", 20)]):
        a.hist(s[col].dropna(), bins=bins, color="#3b6ea5", edgecolor="white")
        a.axvline(s[col].median(), color="crimson", ls="--", lw=1.2, label=f"mediana {s[col].median():.3g}")
        a.set(title=t, xlabel=t.split(" (")[0], ylabel="estudios"); a.legend(fontsize=8)
    fig.suptitle(f"Distribución de parámetros de adquisición (n = {len(s)} estudios)", y=1.02)
    fig.tight_layout(); fig.savefig(out / "01_distribuciones_adquisicion.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_boxplots(s, labels, out):
    d = s.drop(columns=["patient_overall"], errors="ignore").merge(labels, on="StudyInstanceUID", how="left")
    d["grosor"] = d.thickness.round(3).astype(str) + " mm"
    top = d.grosor.value_counts().index[:5]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    d[d.grosor.isin(top)].boxplot(column="n_slices", by="grosor", ax=ax[0], grid=False)
    ax[0].set(title="Cortes por estudio según grosor", xlabel="SliceThickness", ylabel="n.º de cortes")
    d[d.grosor.isin(top)].boxplot(column="pixel_mm", by="grosor", ax=ax[1], grid=False)
    ax[1].set(title="PixelSpacing según grosor", xlabel="SliceThickness", ylabel="mm")
    if d.patient_overall.notna().any():
        d.boxplot(column="n_slices", by="patient_overall", ax=ax[2], grid=False)
        ax[2].set(title="Cortes por estudio según fractura", xlabel="patient_overall (0 = sin, 1 = con)", ylabel="n.º de cortes")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(out / "02_boxplots_comparacion.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_protocolos(s, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ct = pd.crosstab(s.transfer_syntax.str.replace(r" \(.*", "", regex=True), s.intercept.astype(int).astype(str))
    ct.plot.barh(stacked=True, ax=ax[0]); ax[0].set(title="Transfer syntax × RescaleIntercept", xlabel="estudios", ylabel="")
    ax[0].legend(title="intercept", fontsize=8)
    s.groupby(s.rows.astype(str) + "×" + s.cols.astype(str)).size().plot.bar(ax=ax[1], color="#8a6bbf")
    ax[1].set(title="Matriz de la imagen", xlabel="", ylabel="estudios"); ax[1].tick_params(axis="x", rotation=0)
    ax[2].scatter(s.pixel_mm, s.n_slices, c=s.thickness, cmap="viridis", s=14, alpha=.75)
    ax[2].set(title="Resolución en plano vs. n.º de cortes", xlabel="PixelSpacing (mm)", ylabel="cortes")
    fig.colorbar(ax[2].collections[0], ax=ax[2], label="SliceThickness (mm)")
    fig.tight_layout(); fig.savefig(out / "03_protocolos_y_resolucion.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_consistencia(s, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(s.thickness, s.spacing_z, s=12, alpha=.6, color="#2a9d8f")
    lim = [min(s.thickness.min(), s.spacing_z.min()), max(s.thickness.max(), s.spacing_z.max())]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set(title="SliceThickness vs. espaciado z medido", xlabel="SliceThickness (mm)", ylabel="espaciado z (mm)")
    ax[1].hist(s.n_slices * s.spacing_z, bins=30, color="#e76f51", edgecolor="white")
    ax[1].set(title="Cobertura aproximada en z (cortes × espaciado)", xlabel="mm", ylabel="estudios")
    fig.tight_layout(); fig.savefig(out / "04_consistencia_z.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- auditoría del pipeline
def auditoria_pipeline(s, results_dir, out):
    """Cruza los metadatos por estudio con la auditoría del pipeline (solo lectura de CSV, sin píxeles):
    orden_z_auditoria.csv, preprocess_times.csv y preprocess_failures.csv. Devuelve (dict de cifras, DataFrame
    de estudios con orden z anómalo) y escribe out/07_auditoria_pipeline.png. Si falta un CSV, se omite ese panel."""
    results_dir = Path(results_dir)
    res = {}
    fz, tm, fl = (results_dir / n for n in ("orden_z_auditoria.csv", "preprocess_times.csv", "preprocess_failures.csv"))
    z = pd.read_csv(fz) if fz.exists() else None
    t = pd.read_csv(tm) if tm.exists() else None
    if fl.exists():
        res["fallos_preprocesamiento"] = int(len(pd.read_csv(fl)))
    lab = s[[c for c in ("StudyInstanceUID", "patient_overall", "transfer_syntax") if c in s]]
    anom = pd.DataFrame()
    if z is not None:
        res["z_no_monotono"] = int((~z.z_monotonic_vs_instance).sum())
        res["z_direccion"] = z.z_direction_vs_instance.value_counts().to_dict()
        res["dz_cv_mediana"] = round(float(z.dz_cv.median()), 3)
        anom = z[(~z.z_monotonic_vs_instance) | (z.z_direction_vs_instance == "mixto") | (z.dz_cv > 0.25)]
        res["z_anomalos"] = [u.split(".")[-1] for u in anom.StudyInstanceUID]
        if "patient_overall" in lab:
            zz = z.merge(lab, on="StudyInstanceUID")
            ct = pd.crosstab(zz.z_direction_vs_instance.replace({"mixto": "creciente"}), zz.patient_overall)
            res["z_direccion_vs_fractura"] = {str(k): {str(a): int(b) for a, b in v.items()} for k, v in ct.to_dict().items()}
            try:
                from scipy.stats import chi2_contingency
                res["z_direccion_chi2_p"] = float(chi2_contingency(ct)[1])
            except Exception:
                pass
    tt = None
    if t is not None:
        res["tiempo_s"] = {k: round(float(v), 2) for k, v in t.seconds.describe()[["mean", "50%", "max"]].items()}
        if "transfer_syntax" in lab:
            tt = t.merge(lab, on="StudyInstanceUID")
            res["tiempo_mediano_por_transfer_syntax"] = tt.groupby("transfer_syntax").seconds.median().round(2).to_dict()
    n = (2 if z is not None else 0) + (1 if tt is not None else 0)
    if n:
        fig, ax = plt.subplots(1, n, figsize=(5.2 * n, 4)); ax = np.atleast_1d(ax); k = 0
        if z is not None:
            z.dz_cv.clip(upper=0.5).plot.hist(bins=50, ax=ax[k], color="#4C72B0"); ax[k].axvline(0.25, color="r", ls="--", lw=1)
            ax[k].set(title="Irregularidad del espaciado z (dz_cv, recortado a 0.5)", xlabel="dz_cv"); k += 1
            d = z.z_direction_vs_instance.value_counts(); ax[k].bar(d.index, d.values, color="#55A868")
            ax[k].set(title="Sentido de z respecto a InstanceNumber", ylabel="estudios"); k += 1
        if tt is not None:
            g = [(a, d.seconds.values) for a, d in tt.groupby("transfer_syntax")]
            ax[k].boxplot([v for _, v in g], showfliers=False); ax[k].set_xticklabels([a.split(" (")[0][:22] for a, _ in g])
            ax[k].set(title="Tiempo de preprocesamiento por estudio (s)", ylabel="s"); ax[k].tick_params(axis="x", labelsize=7, rotation=20)
        fig.tight_layout(); fig.savefig(Path(out) / "07_auditoria_pipeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return res, anom


# --------------------------------------------------------------------------- informe
def write_report(s, checks, omit, meta, out):
    d = s[["n_slices", "thickness", "pixel_mm", "spacing_z"]].describe().round(3)
    L = [f"# Informe de calidad DICOM — RSNA 2022 (generado por `scripts/eda_metadatos_dicom.py`)", "",
         f"- Fuente: **{meta['fuente']}**", f"- Estudios analizados: **{len(s)}** (semilla {SEED}; {meta['muestreo']})",
         f"- Cortes leídos: {meta['cortes']}", "", "## Estadística descriptiva (por estudio)", "", d.to_markdown(), "",
         "## Comprobaciones de calidad", "", "| Comprobación | Estudios | % | Ejemplos (UID) |", "|---|---:|---:|---|"]
    for k, v in checks.items():
        L.append(f"| {v['descripcion']} | {v['n']} | {v['pct']} | {', '.join(u.split('.')[-1] for u in v['ejemplos']) or '—'} |")
    L += ["", "## Limitaciones", "", f"- {meta['limitacion']}"]
    if omit:
        L.append(f"- Comprobaciones omitidas por no existir la columna en la fuente: {', '.join(omit)}.")
    (out / "INFORME_CALIDAD_DICOM.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dicom-dir"); g.add_argument("--from-audit")
    ap.add_argument("--train-csv", default=str(REPO / "train.csv"))
    ap.add_argument("--n-studies", type=int, default=300, help="solo con --dicom-dir; 0 = todos")
    ap.add_argument("--pipeline-results", default=None, help="carpeta con orden_z_auditoria.csv, preprocess_times.csv y preprocess_failures.csv (opcional)")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "eda_imagenes"))
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(a.train_csv)[["StudyInstanceUID", "patient_overall"]]

    if a.dicom_dir:
        from tqdm import tqdm
        uids = sample_studies(Path(a.train_csv), a.n_studies)
        rows = [r for u in tqdm(uids) for r in read_study_headers(Path(a.dicom_dir) / u)]
        sl = pd.DataFrame(rows)
        errs = sl["error"].notna().sum() if "error" in sl else 0
        sl.drop(columns=["error"], errors="ignore").to_csv(out / "metadata_slices.csv.gz", index=False)
        s, omit = studies_from_headers(sl.dropna(subset=["rows"])), []
        meta = dict(fuente=f"cabeceras DICOM de {a.dicom_dir}", cortes=f"{len(sl)} (ilegibles: {errs})",
                    muestreo="estratificada por patient_overall" if a.n_studies > 0 else "todos los estudios",
                    limitacion="Solo cabeceras; no se evalúa el contenido de píxeles (ver notebook 02).")
    else:
        audit = pd.read_csv(a.from_audit)
        s, omit = studies_from_audit(audit)
        meta = dict(fuente=f"auditoría del pipeline ({a.from_audit})", cortes=f"{len(audit)} (48 centrales por estudio)",
                    muestreo="todos los estudios de la partición v2",
                    limitacion="Solo 48 cortes centrales por estudio: no detecta huecos en z ni cabeceras fuera de esos cortes; "
                               "n_slices proviene de n_files_study. En esta fuente spacing_z es la distancia entre dos cortes "
                               "consecutivos centrales (sample_25d), no una mediana del estudio: la comprobación "
                               "grosor-vs-espaciado z es orientativa y debe confirmarse con --dicom-dir.")
    s.merge(labels, on="StudyInstanceUID", how="left").to_csv(out / "metadata_studies.csv", index=False)  # igual que el notebook: incluye patient_overall
    checks = quality_checks(s)
    (out / "quality_report.json").write_text(json.dumps(dict(meta=meta, n_estudios=len(s), checks=checks,
                                                              checks_omitidos=omit), indent=2, ensure_ascii=False))
    fig_distribuciones(s, out / "figures"); fig_boxplots(s, labels, out / "figures")
    fig_protocolos(s, out / "figures"); fig_consistencia(s, out / "figures")
    if a.pipeline_results:
        aud, _ = auditoria_pipeline(s.merge(labels, on="StudyInstanceUID", how="left"), a.pipeline_results, out / "figures")
        (out / "auditoria_pipeline.json").write_text(json.dumps(aud, indent=2, ensure_ascii=False))
    write_report(s, checks, omit, meta, out)
    print(f"OK -> {out}  ({len(s)} estudios)")


if __name__ == "__main__":
    main()
