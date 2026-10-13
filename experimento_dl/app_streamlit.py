"""App Streamlit — RSNA 2022 · clasificación global de estudios TC de columna cervical (v2).

Ejecutar desde la raíz del repositorio:
    .venv/bin/streamlit run experimento_dl/app_streamlit.py          (Windows: .venv\\Scripts\\streamlit)

- Artefactos: artifacts/v2/ (métricas, predicciones y figuras reales del entrenamiento en Kaggle).
- Checkpoints: artifacts/v2/weights/{mil,bigru}_final.pt (fuera de git). Si faltan, la predicción
  aparece como PENDIENTE; nunca se muestran probabilidades simuladas.
- Preprocesamiento e inferencia: exclusivamente `inference.py` / `s03_preprocess` (vía app_core).
- Las etiquetas reales se muestran separadas y rotuladas; nunca como predicción.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_core as core  # fija RSNA_EXP=v2 antes de importar common
from common import BONE_WL, BONE_WW, IMG_SIZE, TRAIN_CSV, REPO

st.set_page_config(page_title="RSNA 2022 · Fracturas cervicales", page_icon="🦴", layout="wide")

COLORS = {"mil": "#2a6fdb", "bigru": "#e07b39", core.SHORT["mil"]: "#2a6fdb", core.SHORT["bigru"]: "#e07b39"}
PRE = core.preprocessing_config() or {}
K = PRE.get("k_slices", 48)

LIMITS = f"""
**Este sistema NO es una herramienta clínica.**
- Clasificador **global por estudio** (¿hay al menos una fractura cervical?): no detecta ni localiza
  fracturas ni identifica la vértebra. Las probabilidades por posición del modelo MIL son una ayuda de
  interpretación, no una localización validada.
- Entrada reducida a **{K} posiciones 2.5D** por estudio (corte central ± {PRE.get('context_mm', 2.0)} mm);
  el muestreo 5 %–95 % no garantiza incluir los cortes donde la fractura es visible.
- Entrenado y evaluado con los 2 019 estudios de `train.csv` de RSNA 2022 (particiones 64/16/20,
  semilla 42). Desempeño modesto (AUC de prueba 0.64–0.74) con intervalos de confianza amplios.
- Sin validación clínica ni externa; otro escáner, protocolo o población puede degradar el desempeño.
"""


# ----------------------------------------------------------------------------- caché
@st.cache_data(show_spinner=False)
def train_df():
    return pd.read_csv(TRAIN_CSV)


@st.cache_data(show_spinner=False)
def meta_df():
    return core.dicom_metadata()


@st.cache_data(show_spinner="Verificando checkpoints (SHA-256)…")
def ckpt_status(m, mtime):  # mtime invalida la caché si el archivo cambia
    return core.checkpoint_status(m)


def ckpt_status_now(m):
    p = core.checkpoint_path(m)
    return ckpt_status(m, p.stat().st_mtime if p.exists() else None)


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(m, mtime):
    loaded, skipped = core.load_models([m])
    if skipped:
        raise RuntimeError(skipped[m])
    return loaded[m]


@st.cache_data(show_spinner="Leyendo DICOM…", max_entries=3)
def read_study(key, _payload):
    """_payload: tupla de (nombre, bytes) o de rutas; no se hashea (la clave es `key`)."""
    items = [p if isinstance(p, str) else _Named(*p) for p in _payload]
    return core.read_dicoms(items)


class _Named:
    def __init__(self, name, data):
        self.name, self._d = name, data

    def getvalue(self):
        return self._d


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key)


def gauge(p, thr, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p, number={"valueformat": ".3f"}, title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": color},
               "threshold": {"line": {"color": "black", "width": 3}, "thickness": .8, "value": thr}}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def index_slider(label, n, key):
    """Slider de índice que tolera n == 1 (st.slider exige min < max)."""
    if n <= 1:
        st.caption(f"{label}: único corte disponible")
        return 0
    return st.slider(label, 0, n - 1, n // 2, key=key)


def raw_display(hu):
    """Solo visualización del corte crudo: HU recortadas a [-1024, 2000] → [0,1]."""
    return (np.clip(hu, -1024, 2000) + 1024) / 3024


# ----------------------------------------------------------------------------- barra lateral
st.title("Fracturas de columna cervical en TC — RSNA 2022")
st.caption("Proyecto 2 · CC3084 Data Science · UVG — clasificación global por estudio con dos modelos profundos")
st.error("Prototipo académico. No es un dispositivo médico ni tiene validación clínica.", icon="⚠️")

with st.sidebar:
    st.header("Opciones de visualización")
    interactive = st.toggle("Gráficas interactivas (Plotly)", value=True,
                            help="Desactivado: se muestran las figuras PNG estáticas generadas en el entrenamiento.")
    CHARTS = ["Distribución de clases", "Fractura por vértebra", "Co-ocurrencia", "Metadatos DICOM",
              "Histograma HU", "Matriz de confusión", "Curva ROC", "Curvas de entrenamiento",
              "Barras de métricas", "Distribución de probabilidades", "Dispersión A vs B",
              "Probabilidad por posición (MIL)"]
    visible = set(st.multiselect("Gráficas visibles", CHARTS, default=CHARTS,
                                 help="Muestra u oculta cada tipo de gráfica en todas las pestañas."))
    st.divider()
    st.header("Estado de los modelos")
    STATUS = {m: ckpt_status_now(m) for m in core.MODELS}
    for m in core.MODELS:
        s = STATUS[m]
        icon = {"DISPONIBLE": "✅", "PENDIENTE": "⏳"}.get(s["estado"], "⛔")
        st.markdown(f"{icon} **{core.SHORT[m]}** — checkpoint {s['estado']}")
    met_ok = [m for m in core.MODELS if core.metrics(m) is not None]
    st.caption(f"Métricas reales disponibles: {', '.join(core.SHORT[m] for m in met_ok) or 'ninguna'}")
    st.caption(f"Entrada: K = {K} posiciones 2.5D · ventana ósea L{BONE_WL}/W{BONE_WW} · {IMG_SIZE}×{IMG_SIZE}")

tabs = st.tabs(["1 · Datos y EDA", "2 · Preprocesamiento", "3 · Visor DICOM", "4 · Predicción",
                "5 · Métricas", "6 · Comparación", "7 · Estado e integración", "8 · Limitaciones"])

# 1 · Datos y EDA --------------------------------------------------------------
with tabs[0]:
    train = train_df()
    verts = [f"C{i}" for i in range(1, 8)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudios (train.csv)", f"{len(train):,}")
    c2.metric("Con ≥1 fractura", f"{train.patient_overall.mean():.1%}")
    c3.metric("Vértebra más afectada", train[verts].mean().idxmax())
    c4.metric("Fracturas múltiples", f"{(train[verts].sum(axis=1) > 1).mean():.1%}",
              help="Estudios con más de una vértebra fracturada")
    if interactive:
        a, b = st.columns(2)
        if "Distribución de clases" in visible:
            vc = train.patient_overall.map({0: "sin fractura", 1: "con fractura"}).value_counts().reset_index()
            with a:
                show(px.bar(vc, x="patient_overall", y="count", color="patient_overall", text="count",
                            title="Estudios por clase (patient_overall)",
                            labels={"patient_overall": "", "count": "estudios"}))
        if "Fractura por vértebra" in visible:
            sel = b.radio("Población", ["Todos los estudios", "Solo estudios con fractura"], horizontal=True)
            base = train if sel.startswith("Todos") else train[train.patient_overall == 1]
            fv = (base[verts].mean() * 100).reset_index()
            fv.columns = ["vértebra", "% fracturada"]
            with b:
                show(px.bar(fv, x="vértebra", y="% fracturada", text_auto=".1f",
                            title=f"Prevalencia de fractura por vértebra · {sel.lower()}"))
        if "Co-ocurrencia" in visible:
            n_v = train[verts].sum(axis=1).value_counts().sort_index().reset_index()
            n_v.columns = ["vértebras fracturadas", "estudios"]
            a, b = st.columns(2)
            with a:
                show(px.imshow(train[verts].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                               zmin=-1, zmax=1, title="Correlación de fracturas entre vértebras"))
            with b:
                show(px.bar(n_v, x="vértebras fracturadas", y="estudios", text="estudios",
                            title="Número de vértebras fracturadas por estudio"))
    else:
        for f in sorted((REPO / "imgs").glob("*.png"))[:4]:
            st.image(str(f), caption=f.name)

    sp = core.read_csv(core.RESULTS_DIR / "splits_summary.csv")
    if sp is not None:
        st.subheader("Particiones del experimento v2 (por estudio, estratificadas, semilla 42)")
        st.dataframe(sp.rename(columns={"n": "estudios", "prop_pos": "proporción positivos"}),
                     hide_index=True)

    md = meta_df()
    if md is not None and "Metadatos DICOM" in visible:
        st.subheader("Exploración de metadatos DICOM (cortes centrales usados por el modelo)")
        per_study = md.groupby("StudyInstanceUID").agg(
            cortes=("n_files_study", "first"), grosor_mm=("slice_thickness", "first"),
            espaciado_z_mm=("spacing_z_mm", "first"), slope=("slope", "first"),
            intercept=("intercept", "first"), transfer_syntax=("transfer_syntax", "first"),
            filas=("rows", "first")).reset_index()
        per_study = per_study.merge(train[["StudyInstanceUID", "patient_overall"]], how="left")
        per_study["clase"] = per_study.patient_overall.map({0: "sin fractura", 1: "con fractura"})
        st.caption(f"{len(md):,} cortes auditados de {len(per_study):,} estudios "
                   "(fuente: artifacts/v2/results/dicom_metadata.csv.gz)")
        var = st.selectbox("Variable", ["cortes", "grosor_mm", "espaciado_z_mm", "filas"],
                           format_func={"cortes": "Número de cortes del estudio", "grosor_mm": "Grosor de corte (mm)",
                                        "espaciado_z_mm": "Espaciado en z (mm)", "filas": "Filas (píxeles)"}.get)
        by_class = st.checkbox("Separar por clase", value=True)
        a, b = st.columns([3, 2])
        if interactive:
            with a:
                show(px.histogram(per_study, x=var, color="clase" if by_class else None, nbins=40,
                                  barmode="overlay", opacity=.7, title=f"Distribución de {var} por estudio"))
            ts = per_study.transfer_syntax.value_counts().reset_index()
            with b:
                show(px.pie(ts, names="transfer_syntax", values="count", hole=.4,
                            title="Codificación (transfer syntax)"))
        with st.expander("Combinaciones RescaleSlope / Intercept (control de calidad de HU)"):
            st.dataframe(per_study.groupby(["slope", "intercept"]).size().rename("estudios").reset_index(),
                         hide_index=True)

    with st.expander("Figuras del EDA tabular original (carpeta imgs/)"):
        for f in sorted((REPO / "imgs").glob("*.png")):
            st.image(str(f), caption=f.name)

# 2 · Preprocesamiento ---------------------------------------------------------
with tabs[1]:
    st.markdown(f"""
Mismo preprocesamiento en entrenamiento, evaluación y en esta app (`s03_preprocess` vía `inference.py`):

1. Lectura con **pydicom** (JPEG Lossless vía pylibjpeg/gdcm).
2. **HU = píxel × RescaleSlope + RescaleIntercept** (por archivo); PixelPaddingValue → aire; MONOCHROME1 se invierte.
3. **Ventana ósea** L{BONE_WL}/W{BONE_WW} → [{BONE_WL - BONE_WW // 2}, {BONE_WL + BONE_WW // 2}] HU → [0, 1].
4. Redimensión a **{IMG_SIZE}×{IMG_SIZE}** (INTER_AREA).
5. **{K} posiciones** equiespaciadas entre el 5 % y el 95 % de las instancias.
6. **2.5D**: cada posición aporta 3 canales (corte central y vecinos a ±{PRE.get('context_mm', 2.0)} mm).
7. Orden anatómico por **ImagePositionPatient[2]** (z). Tensor final `[{K}, 3, {IMG_SIZE}, {IMG_SIZE}]`.
""")
    with st.expander("config/preprocessing.json (publicado por entrenamiento)"):
        st.json(PRE)
    kinds = {"Entrada 2.5D": "entrada_25d_", "Antes / después": "preproc_antes_despues_",
             "Montaje de cortes": "montaje_cortes_"}
    prefix = kinds[st.radio("Figura", list(kinds), horizontal=True)]
    figs = core.figures(prefix)
    if figs:
        f = st.selectbox("Estudio de ejemplo", figs, format_func=lambda p, pre=prefix: p.stem.replace(pre, ""),
                         help="y0 = sin fractura, y1 = con fractura (etiqueta real)")
        st.image(str(f), width="stretch")
    else:
        st.info("PENDIENTE: no hay figuras de preprocesamiento en artifacts/v2/results/figures/.")

# 3 · Visor DICOM --------------------------------------------------------------
def study_selector(key):
    """Selector de estudio (local o subido). Devuelve (datasets, errores, avisos) o None."""
    local = core.local_studies()
    src = st.radio("Fuente", ["Subir archivos .dcm", "Estudio local (experimento_dl/data/dicom/)"],
                   horizontal=True, key=f"{key}_src")
    try:
        if src.startswith("Subir"):
            up = st.file_uploader("Todos los .dcm de UN estudio axial de TC", type=["dcm"],
                                  accept_multiple_files=True, key=f"{key}_up")
            if not up:
                return None
            payload = tuple((f.name, f.getvalue()) for f in up)
            return read_study((key, tuple((f.name, f.size) for f in up)), payload)
        if not local:
            st.info("No hay estudios en `experimento_dl/data/dicom/<StudyInstanceUID>/*.dcm`. "
                    "Sube archivos o copia ahí un estudio descargado de Kaggle.")
            return None
        uid = st.selectbox("Estudio", local, key=f"{key}_uid")
        return read_study((key, uid), tuple(str(p) for p in core.local_study_files(uid)))
    except core.InvalidInput as e:
        st.error(f"Entrada inválida: {e}")
        return None


with tabs[2]:
    st.markdown("Explora un estudio DICOM real con las mismas funciones de preprocesamiento del pipeline.")
    res = study_selector("viewer")
    if res:
        dsets, errors, warns = res
        if errors:
            st.warning(f"{len(errors)} archivo(s) descartados: " + "; ".join(errors[:5]))
        for w in warns:
            st.warning(w)
        dsets = core.sort_by_z(dsets)
        st.success(f"{len(dsets)} cortes legibles, ordenados por ImagePositionPatient (z).")
        with st.expander("Resumen del estudio", expanded=True):
            st.table(pd.DataFrame(core.study_summary(dsets).items(), columns=["campo", "valor"]))
        i = index_slider("Corte (orden z)", len(dsets), "viewer_slice")
        hu, win = core.display_slice(dsets[i])
        c1, c2, c3 = st.columns(3)
        c1.image(raw_display(hu), clamp=True, width="stretch",
                 caption=f"ANTES: HU crudo {hu.shape[0]}×{hu.shape[1]} (rango [-1024, 2000])")
        c2.image(win, clamp=True, width="stretch",
                 caption=f"DESPUÉS: ventana ósea L{BONE_WL}/W{BONE_WW}, [0,1], {IMG_SIZE}×{IMG_SIZE}")
        if "Histograma HU" in visible:
            h = np.histogram(np.clip(hu, -2100, 3100), bins=120)
            fig = go.Figure(go.Bar(x=h[1][:-1], y=h[0], marker_color="gray"))
            fig.add_vrect(x0=BONE_WL - BONE_WW / 2, x1=BONE_WL + BONE_WW / 2, fillcolor="orange",
                          opacity=.25, line_width=0)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log",
                              title="Histograma HU (ventana ósea en naranja)", xaxis_title="HU")
            with c3:
                show(fig)
        with st.expander("Metadatos DICOM del corte"):
            st.json({k: str(dsets[i].get(k, "—")) for k in
                     ["InstanceNumber", "ImagePositionPatient", "SliceThickness", "PixelSpacing", "RescaleSlope",
                      "RescaleIntercept", "PhotometricInterpretation", "BitsStored", "Rows", "Columns"]})
        st.subheader("Comparación de cortes")
        sel = st.multiselect("Cortes a comparar (orden z)", list(range(len(dsets))),
                             default=sorted({0, len(dsets) // 2, len(dsets) - 1}), max_selections=6)
        if sel:
            for c, j in zip(st.columns(len(sel)), sel):
                c.image(core.display_slice(dsets[j])[1], clamp=True, width="stretch",
                        caption=f"corte {j} · z={float(dsets[j].ImagePositionPatient[2]):.1f}"
                        if "ImagePositionPatient" in dsets[j] else f"corte {j}")
        st.subheader(f"Entrada real del modelo ({K} posiciones 2.5D)")
        if st.toggle("Calcular la entrada del modelo (inference.preprocess_study)", key="viewer_input"):
            try:
                vol, n_files, pw = core.model_input(dsets)
                for w in pw:
                    st.warning(w)
                st.caption(f"Tensor uint8 {list(vol.shape)} a partir de {n_files} archivos.")
                j = index_slider("Posición", len(vol), "viewer_pos")
                for c, ch, name in zip(st.columns(3), range(3), ["vecino inferior (−z)", "corte central",
                                                                 "vecino superior (+z)"]):
                    c.image(vol[j, ch], clamp=True, width="stretch", caption=f"canal {ch}: {name}")
                with st.expander("Montaje de las posiciones (canal central)"):
                    cols = st.columns(8)
                    for jj in range(len(vol)):
                        cols[jj % 8].image(vol[jj, 1], clamp=True, width="stretch", caption=str(jj))
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo construir la entrada: {type(e).__name__}: {e}")

# 4 · Predicción ---------------------------------------------------------------
with tabs[3]:
    st.subheader("Predicción sobre un estudio nuevo")
    choice = st.radio("Modelo", [core.NAMES["mil"], core.NAMES["bigru"], "Ambos (comparar)"], index=1,
                      horizontal=True, help="Modelo B (BiGRU) es el recomendado por el equipo de entrenamiento.")
    which = [m for m in core.MODELS if core.NAMES[m] == choice] or list(core.MODELS)
    ready = [m for m in which if STATUS[m]["estado"] == "DISPONIBLE"]
    for m in which:
        if m not in ready:
            st.warning(f"**{core.NAMES[m]}: predicción PENDIENTE** — checkpoint `{STATUS[m]['ruta']}` "
                       f"{'no encontrado' if STATUS[m]['estado'] == 'PENDIENTE' else 'no coincide con MANIFEST.json'}. "
                       "Lo produce el entrenamiento (Majo, Output del notebook de Kaggle). "
                       "No se muestran probabilidades simuladas.")
    res = study_selector("pred")
    if res and ready:
        dsets, errors, warns = res
        if errors:
            st.warning(f"{len(errors)} archivo(s) descartados: " + "; ".join(errors[:5]))
        for w in warns:
            st.warning(w)
        if st.button("Ejecutar predicción", type="primary"):
            results = {}
            for m in ready:
                p = core.checkpoint_path(m)
                try:
                    clf = load_model(m, p.stat().st_mtime)
                    with st.spinner(f"Infiriendo con {core.SHORT[m]}…"):
                        results[m] = core.predict(dsets, {m: clf})[m]
                except Exception as e:  # noqa: BLE001
                    st.error(f"{core.NAMES[m]}: error de inferencia — {type(e).__name__}: {e}")
            st.session_state["pred_results"] = results
        results = st.session_state.get("pred_results", {})
        results = {m: r for m, r in results.items() if m in ready}
        if results:
            for c, (m, r) in zip(st.columns(len(results)), results.items()):
                with c:
                    show(gauge(r["probability"], r["threshold"], core.NAMES[m], COLORS[m]), key=f"g_{m}")
                    lab = "FRACTURA (≥ umbral)" if r["prediction"] else "sin fractura (< umbral)"
                    st.markdown(f"**Predicción del modelo:** {lab}  \nUmbral (Youden en validación): "
                                f"{r['threshold']:.3f} · entrada {r['input_shape']} · {r['seconds']:.1f} s")
                    for w in r["warnings"]:
                        st.caption(f"⚠️ {w}")
            st.info(next(iter(results.values()))["disclaimer"])
            sp = results.get("mil", {}).get("slice_probabilities")
            if sp and "Probabilidad por posición (MIL)" in visible:
                show(px.bar(x=list(range(len(sp))), y=sp, labels={"x": "posición (orden z)", "y": "p"},
                            range_y=[0, 1], title="MIL: probabilidad por posición (interpretación, no localización)"))
    elif res and not ready:
        st.info("El estudio se leyó correctamente; la predicción se habilitará cuando existan los checkpoints.")

    st.divider()
    st.subheader("Predicciones guardadas del conjunto de prueba (n=404)")
    st.caption("Salidas reales de la evaluación en Kaggle (`test_predictions_*.csv`). Están disponibles "
               "sin checkpoints y permiten comparar ambos modelos estudio por estudio.")
    pa, pb = core.predictions("mil"), core.predictions("bigru")
    if pa is None or pb is None:
        st.info("PENDIENTE: faltan test_predictions_{mil,bigru}.csv en artifacts/v2/results/.")
    else:
        both = pa.merge(pb[["StudyInstanceUID", "prob", "pred"]], on="StudyInstanceUID", suffixes=("_mil", "_bigru"))
        flt = st.radio("Filtrar", ["Todos", "Discrepan A vs B", "Ambos aciertan", "Ambos fallan"], horizontal=True)
        ok_a = both.pred_mil == both.patient_overall
        ok_b = both.pred_bigru == both.patient_overall
        both = both[{"Todos": ok_a | ~ok_a, "Discrepan A vs B": both.pred_mil != both.pred_bigru,
                     "Ambos aciertan": ok_a & ok_b, "Ambos fallan": ~ok_a & ~ok_b}[flt]]
        st.caption(f"{len(both)} estudios")
        if len(both):
            uid = st.selectbox("Estudio de prueba", both.StudyInstanceUID)
            row = both.set_index("StudyInstanceUID").loc[uid]
            c = st.columns(3)
            c[0].metric(core.SHORT["mil"], f"{row.prob_mil:.3f}", "fractura" if row.pred_mil else "sin fractura",
                        delta_color="off")
            c[1].metric(core.SHORT["bigru"], f"{row.prob_bigru:.3f}",
                        "fractura" if row.pred_bigru else "sin fractura", delta_color="off")
            c[2].metric("Etiqueta real (NO es predicción)", int(row.patient_overall))
            sp = core.mil_test_slice_probs()
            if sp and uid in sp[0] and "Probabilidad por posición (MIL)" in visible and interactive:
                show(px.bar(x=list(range(sp[1].shape[1])), y=sp[1][sp[0].index(uid)], range_y=[0, 1],
                            labels={"x": "posición (orden z)", "y": "p"},
                            title="MIL: probabilidad por posición guardada (interpretación, no localización)"),
                     key="saved_sp")

# 5 · Métricas -----------------------------------------------------------------
with tabs[4]:
    if not met_ok:
        st.info("PENDIENTE: no hay metrics_*.json en artifacts/v2/results/.")
    set_key = st.radio("Conjunto", list(core.SETS), format_func=core.SETS.get, index=1, horizontal=True)
    for m in met_ok:
        mt = core.metrics(m)
        t = mt[set_key]
        st.subheader(core.NAMES[m])
        for c, k in zip(st.columns(6), core.METRIC_KEYS):
            c.metric(core.METRIC_LABELS[k], f"{t[k]:.3f}")
        lo, hi = mt["test_auc_ci95_bootstrap"]
        st.caption(f"n={t['n']} · umbral {t['threshold']:.3f} · AUC prueba IC95 % bootstrap [{lo:.3f}, {hi:.3f}] · "
                   f"mejor época {mt['best_epoch']}/{mt['epochs_run']} · entrenamiento {mt['train_time_s']:.0f} s "
                   f"({mt['device']}) · inferencia {mt['inference_ms_per_study']:.1f} ms/estudio")
        if interactive:
            cols = st.columns(3)
            if "Matriz de confusión" in visible:
                cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
                with cols[0]:
                    show(px.imshow(cm, text_auto=True, x=["pred 0", "pred 1"], y=["real 0", "real 1"],
                                   color_continuous_scale="Blues", title="Matriz de confusión"), key=f"cm_{m}")
            if "Curva ROC" in visible:
                pr = core.predictions(m, "val" if set_key == "val" else "test")
                if pr is not None:
                    roc, auc = core.roc_points(pr.patient_overall, pr.prob)
                    fig = px.line(roc, x="fpr", y="tpr", hover_data=["threshold"], title=f"ROC (AUC={auc:.3f})",
                                  labels={"fpr": "1 − especificidad", "tpr": "sensibilidad"},
                                  color_discrete_sequence=[COLORS[m]])
                    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                    showlegend=False)
                    with cols[1]:
                        show(fig, key=f"roc_{m}")
            if "Curvas de entrenamiento" in visible:
                h = core.history(m)
                if h is not None:
                    fig = px.line(h, x="epoch", y=["train_loss", "val_loss", "val_auc"], markers=True,
                                  title="Curvas de entrenamiento")
                    fig.add_vline(x=mt["best_epoch"], line_dash="dot", annotation_text="época elegida")
                    with cols[2]:
                        show(fig, key=f"hist_{m}")
        else:
            for c, kind in zip(st.columns(3), ("confusion", "roc", "curvas")):
                f = core.FIG_DIR / f"{kind}_{m}.png"
                if f.exists():
                    c.image(str(f))
        with st.expander("JSON completo de métricas"):
            st.json(mt)

# 6 · Comparación --------------------------------------------------------------
with tabs[5]:
    if len(met_ok) < 2:
        st.info("La comparación requiere las métricas de ambos modelos (PENDIENTE).")
    else:
        long = core.metrics_long()
        a, b = st.columns([1, 2])
        sets_sel = a.multiselect("Conjuntos", list(core.SETS.values()), default=[core.SETS["test_thr_val_youden"]])
        mets_sel = b.multiselect("Métricas", list(core.METRIC_LABELS.values()),
                                 default=list(core.METRIC_LABELS.values()))
        sub = long[long.conjunto.isin(sets_sel) & long.metrica.isin(mets_sel)]
        if interactive and "Barras de métricas" in visible and len(sub):
            show(px.bar(sub, x="metrica", y="valor", color="modelo", barmode="group", facet_row="conjunto"
                        if len(sets_sel) > 1 else None, range_y=[0, 1], text_auto=".3f",
                        color_discrete_map=COLORS, height=330 * max(1, len(sets_sel)),
                        title="Comparación de métricas"))
        if len(sub):
            st.dataframe(sub.pivot_table(index=["conjunto", "metrica"], columns="modelo", values="valor")
                         .assign(**{"Δ (B − A)": lambda d: d[core.SHORT["bigru"]] - d[core.SHORT["mil"]]})
                         .round(3), width="stretch")
        elif not interactive:
            for f in ["comparacion_metricas_prueba.png", "comparacion_roc_prueba.png", "comparacion_tiempos.png"]:
                if (core.FIG_DIR / f).exists():
                    st.image(str(core.FIG_DIR / f))

        pc = core.paired_comparison()
        if pc:
            st.subheader("Pruebas pareadas (BiGRU − MIL)")
            rows = []
            for s in ("val", "test"):
                d = pc[s]
                rows.append({"conjunto": "validación" if s == "val" else "prueba", "n": d["n"],
                             "AUC A (MIL)": d["delong"]["auc_a"], "AUC B (BiGRU)": d["delong"]["auc_b"],
                             "Δ AUC": d["delong"]["diff"], "DeLong p": d["delong"]["p_value"],
                             "IC95 % Δ (bootstrap)": "[{:.3f}, {:.3f}]".format(*d["bootstrap_pareado"]["ci95"]),
                             "McNemar p": d["mcnemar_umbral_val"]["p_value_exacto"],
                             "Spearman probs": d["correlacion_probs_spearman"]})
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
            st.caption(pc.get("nota", ""))

        st.subheader("Exploración interactiva del umbral (conjunto de prueba)")
        st.caption("Recalcula las métricas sobre las predicciones reales guardadas. Es exploratorio: los "
                   "umbrales oficiales se fijaron en validación y no deben reajustarse con la prueba.")
        pa, pb = core.predictions("mil"), core.predictions("bigru")
        if pa is not None and pb is not None:
            thr = st.slider("Umbral", 0.0, 1.0, 0.5, 0.01)
            rows = [dict(modelo=core.SHORT[m], **core.metrics_at_threshold(p.patient_overall, p.prob, thr))
                    for m, p in (("mil", pa), ("bigru", pb))]
            st.dataframe(pd.DataFrame(rows)[["modelo", "tp", "fp", "tn", "fn", "sensitivity", "specificity",
                                              "precision", "f1", "accuracy"]].round(3), hide_index=True)
            if interactive:
                c1, c2 = st.columns(2)
                if "Curva ROC" in visible:
                    fig = go.Figure()
                    for m, p in (("mil", pa), ("bigru", pb)):
                        roc, auc = core.roc_points(p.patient_overall, p.prob)
                        fig.add_scatter(x=roc.fpr, y=roc.tpr, mode="lines", line_color=COLORS[m],
                                        name=f"{core.SHORT[m]} · AUC {auc:.3f}")
                        r = core.metrics_at_threshold(p.patient_overall, p.prob, thr)
                        fig.add_scatter(x=[1 - r["specificity"]], y=[r["sensitivity"]], mode="markers",
                                        marker=dict(size=12, color=COLORS[m], line=dict(width=2, color="black")),
                                        showlegend=False)
                    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                    showlegend=False)
                    fig.update_layout(title="ROC en prueba (punto = umbral elegido)", xaxis_title="1 − especificidad",
                                      yaxis_title="sensibilidad", legend=dict(x=.55, y=.05))
                    with c1:
                        show(fig, key="roc_cmp")
                if "Distribución de probabilidades" in visible:
                    d = pd.concat([pa.assign(modelo=core.SHORT["mil"]), pb.assign(modelo=core.SHORT["bigru"])])
                    d["clase real"] = d.patient_overall.map({0: "sin fractura", 1: "con fractura"})
                    fig = px.box(d, x="modelo", y="prob", color="clase real", points="outliers",
                                 title="Probabilidades por clase real")
                    fig.add_hline(y=thr, line_dash="dot")
                    with c2:
                        show(fig, key="box_cmp")
                if "Dispersión A vs B" in visible:
                    both = pa.merge(pb[["StudyInstanceUID", "prob"]], on="StudyInstanceUID", suffixes=("_mil", "_bigru"))
                    both["clase real"] = both.patient_overall.map({0: "sin fractura", 1: "con fractura"})
                    fig = px.scatter(both, x="prob_mil", y="prob_bigru", color="clase real", opacity=.7,
                                     hover_data=["StudyInstanceUID"], title="Probabilidad por estudio: A vs B",
                                     labels={"prob_mil": "p · A (MIL)", "prob_bigru": "p · B (BiGRU)"})
                    fig.add_hline(y=thr, line_dash="dot"); fig.add_vline(x=thr, line_dash="dot")
                    show(fig, key="scatter_cmp")
        rep = core.RESULTS_DIR / "INFORME_RESULTADOS.md"
        if rep.exists():
            with st.expander("Informe de resultados generado en Kaggle"):
                st.markdown(rep.read_text(encoding="utf-8"))

# 7 · Estado e integración -----------------------------------------------------
with tabs[6]:
    st.subheader("Artefactos que consume la app")
    st.dataframe(core.artifact_status(), hide_index=True, width="stretch")
    st.subheader("Checkpoints")
    st.dataframe(pd.DataFrame([{k: v for k, v in STATUS[m].items() if k != "model"} for m in core.MODELS]),
                 hide_index=True, width="stretch")
    st.markdown("""
Para habilitar la predicción, copiar `mil_final.pt` y `bigru_final.pt` (Output del notebook de Kaggle)
en `artifacts/v2/weights/`. La app verifica su SHA-256 contra `artifacts/v2/MANIFEST.json` y solo usa
los archivos que coinciden. Contrato: `experimento_dl/INTEGRACION_STREAMLIT.md`.
""")

# 8 · Limitaciones -------------------------------------------------------------
with tabs[7]:
    st.markdown(LIMITS)
