"""App Streamlit del experimento RSNA 2022 (muestra reducida).

Ejecutar desde experimento_dl/:   ../.venv/Scripts/streamlit run app_streamlit.py

- El preprocesamiento de estudios subidos usa EXACTAMENTE las funciones de s03_preprocess
  (slice_record, build_volume, select_uploaded) y la normalización de s04_train_eval (to_input).
- Solo se ofrecen modelos con pesos (data/weights/*_best.pt) y métricas (results/metrics_*.json).
- Las etiquetas reales se muestran separadas y rotuladas; nunca como predicción.
"""
import io
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (BONE_WL, BONE_WW, CACHE_DIR, FIG_DIR, IMG_SIZE, K_SLICES, REPO, RESULTS_DIR,
                    TRAIN_CSV, WEIGHTS_DIR)

st.set_page_config(page_title="RSNA 2022 · clasificación de estudios TC", layout="wide")

NAMES = {"mil": "ResNet18 + MIL (top-3)", "bigru": "ResNet18 + BiGRU"}
LIMITS = """
**Este sistema NO es una herramienta clínica.**
- Clasificador **global por estudio** (¿hay al menos una fractura cervical?): no detecta ni localiza
  fracturas ni identifica la vértebra afectada. Las probabilidades por corte del modelo MIL son
  una ayuda de interpretación, no una localización validada.
- Entrenado con una **muestra pequeña** de RSNA 2022 y solo **24 cortes por estudio**; el
  muestreo no garantiza incluir los cortes donde la fractura es visible.
- Evaluado en un conjunto de prueba pequeño (ver IC95 % del AUC): las métricas son inestables.
- Sin validación clínica ni externa; distinto escáner/protocolo puede degradar el desempeño.
"""


# ----------------------------------------------------------------------------- utilidades
def read_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def available_models():
    out = {}
    for m in NAMES:
        met = read_json(RESULTS_DIR / f"metrics_{m}.json")
        w = WEIGHTS_DIR / f"{m}_best.pt"
        if met is not None:
            out[m] = dict(metrics=met, weights=w if w.exists() else None)
    return out


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(name):
    import torch
    from s04_train_eval import MILResNet, ResNetBiGRU
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (MILResNet() if name == "mil" else ResNetBiGRU()).to(dev)
    model.load_state_dict(torch.load(WEIGHTS_DIR / f"{name}_best.pt", map_location=dev))
    model.eval()
    return model, dev


def predict_volume(name, vol_uint8):
    """vol_uint8: [K,224,224] uint8 (salida de build_volume). Devuelve prob de estudio y por corte."""
    import torch
    from s04_train_eval import to_input
    model, dev = load_model(name)
    x = torch.from_numpy(vol_uint8[None])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        study_logit, slice_logits = model(to_input(x, dev, train=False))
    p = float(torch.sigmoid(study_logit.float())[0])
    ps = None if slice_logits is None else torch.sigmoid(slice_logits.float())[0].cpu().numpy()
    return p, ps


def preprocess_uploaded(files):
    import pydicom
    from s03_preprocess import build_volume, select_uploaded, slice_record
    dsets, bad = [], []
    for f in files:
        try:
            dsets.append(pydicom.dcmread(io.BytesIO(f.getvalue())))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{f.name}: {type(e).__name__}")
    if not dsets:
        return None, None, bad
    sel = select_uploaded(dsets)
    recs = [slice_record(ds, f"inst_{getattr(ds, 'InstanceNumber', i)}", "subido", i + 1)
            for i, ds in enumerate(sel)]
    vol, recs = build_volume(recs)
    meta = pd.DataFrame([r["meta"] for r in recs]).drop(columns=["StudyInstanceUID"])
    return vol, meta, bad


def show_volume(vol, key, slice_probs=None):
    i = st.slider("Corte (ordenado por z / ImagePositionPatient)", 0, len(vol) - 1, len(vol) // 2, key=key)
    c1, c2 = st.columns([1, 1])
    c1.image(vol[i], width=340, clamp=True,
             caption=f"corte {i} · ventana ósea L{BONE_WL}/W{BONE_WW} · {IMG_SIZE}×{IMG_SIZE}")
    if slice_probs is not None:
        fig = go.Figure(go.Bar(x=list(range(len(slice_probs))), y=slice_probs,
                               marker_color=["#c0392b" if j == i else "#2a6fdb" for j in range(len(slice_probs))]))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                          title="MIL: probabilidad por corte (interpretación, no localización)",
                          xaxis_title="corte (orden z)", yaxis_title="p", yaxis_range=[0, 1])
        c2.plotly_chart(fig, use_container_width=True)


def gauge(p, thr, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p, number={"valueformat": ".3f"}, title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#2a6fdb"},
               "threshold": {"line": {"color": "black", "width": 3}, "value": thr}}))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ----------------------------------------------------------------------------- interfaz
st.title("Clasificación global de estudios TC de columna cervical — RSNA 2022")
st.error("Prototipo académico. No es un dispositivo médico ni tiene validación clínica.", icon="⚠️")

models = available_models()
with st.sidebar:
    st.header("Opciones")
    interactive = st.toggle("Gráficos interactivos", value=True)
    if models:
        st.success("Modelos evaluados: " + ", ".join(NAMES[m] for m in models))
    else:
        st.warning("Todavía no hay modelos entrenados.")
    st.caption(f"K = {K_SLICES} cortes por estudio · ventana ósea L{BONE_WL}/W{BONE_WW}")

tabs = st.tabs(["1 · Dataset y EDA", "2 · Preprocesamiento", "3 · Predicción", "4 · Métricas",
                "5 · Comparación", "6 · Limitaciones"])

# 1 · Dataset y EDA ------------------------------------------------------------
with tabs[0]:
    train = pd.read_csv(TRAIN_CSV)
    verts = [f"C{i}" for i in range(1, 8)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Estudios en train.csv", f"{len(train):,}")
    c2.metric("Con fractura (patient_overall=1)", f"{train.patient_overall.mean():.1%}")
    c3.metric("Estudios con bounding boxes", "235")
    if interactive:
        a, b = st.columns(2)
        vc = train.patient_overall.map({0: "sin fractura", 1: "con fractura"}).value_counts().reset_index()
        a.plotly_chart(px.bar(vc, x="patient_overall", y="count", title="patient_overall",
                              color="patient_overall"), use_container_width=True)
        fv = (train[verts].mean() * 100).reset_index(); fv.columns = ["vértebra", "% fracturada"]
        b.plotly_chart(px.bar(fv, x="vértebra", y="% fracturada", title="Fractura por vértebra (%)"),
                       use_container_width=True)
        st.plotly_chart(px.imshow(train[verts].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Co-ocurrencia de fracturas entre vértebras"),
                        use_container_width=True)
    else:
        for f in sorted((REPO / "imgs").glob("*.png"))[:4]:
            st.image(str(f), caption=f.name)
    with st.expander("Figuras del EDA original (carpeta imgs/)"):
        for f in sorted((REPO / "imgs").glob("*.png")):
            st.image(str(f), caption=f.name)
    sp = RESULTS_DIR / "splits.csv"
    if sp.exists():
        st.subheader("Muestra y particiones del experimento (por estudio)")
        s = pd.read_csv(sp)
        st.dataframe(s.groupby("split")["patient_overall"].agg(estudios="size", positivos="sum"))
    dl = RESULTS_DIR / "download_log.csv"
    if dl.exists():
        d = pd.read_csv(dl)
        ok = d[d.status == "ok"]
        st.caption(f"Estudios descargados: {len(ok)} · {ok['mb'].sum() / 1e3:.2f} GB · "
                   f"N cortes por estudio (mediana): {int(ok['n_slices'].median())}")
        if interactive and len(ok):
            st.plotly_chart(px.histogram(ok, x="n_slices", nbins=30, title="Número total de cortes por estudio"),
                            use_container_width=True)

# 2 · Preprocesamiento ---------------------------------------------------------
with tabs[1]:
    st.markdown(f"""
1. Lectura con **pydicom** (incluye JPEG Lossless vía pylibjpeg/gdcm).
2. **HU = pixel × RescaleSlope + RescaleIntercept** (leídos de cada archivo); PixelPaddingValue → aire.
3. **MONOCHROME1** se invierte; MONOCHROME2 se usa tal cual.
4. **Ventana ósea** L{BONE_WL}/W{BONE_WW} → rango [{BONE_WL - BONE_WW // 2}, {BONE_WL + BONE_WW // 2}] HU → **[0, 1]**.
5. Redimensión a **{IMG_SIZE}×{IMG_SIZE}** (INTER_AREA).
6. Orden por **ImagePositionPatient[2]** (z), no por número de instancia.
7. **{K_SLICES} cortes** equiespaciados entre el 5 % y el 95 % de las instancias.
""")
    figs = sorted(FIG_DIR.glob("preproc_antes_despues_*.png"))
    if figs:
        st.image(str(st.selectbox("Ejemplo antes/después", figs, format_func=lambda p: p.name)))
    mont = sorted(FIG_DIR.glob("montaje_cortes_*.png"))
    if mont:
        st.image(str(st.selectbox("Montaje de cortes", mont, format_func=lambda p: p.name)))
    cached = sorted(CACHE_DIR.glob("*.npy"))
    if cached:
        st.subheader("Volúmenes preprocesados locales")
        f = st.selectbox("Estudio en caché", cached, format_func=lambda p: p.stem)
        show_volume(np.load(f), key="cache_slider")
    aud = RESULTS_DIR / "orden_z_auditoria.csv"
    if aud.exists():
        with st.expander("Auditoría del orden z y metadatos DICOM"):
            st.dataframe(pd.read_csv(aud))
            m = pd.read_csv(RESULTS_DIR / "dicom_metadata.csv")
            st.write(m[["slope", "intercept", "photometric", "transfer_syntax"]].value_counts().rename("cortes"))

# 3 · Predicción ---------------------------------------------------------------
with tabs[2]:
    ready = {m: v for m, v in models.items() if v["weights"] is not None}
    if not ready:
        st.info("**Predicción no disponible todavía**: no hay modelos entrenados con pesos locales "
                "(data/weights/*_best.pt). No se muestran predicciones.")
    else:
        choice = st.radio("Modelo", [NAMES[m] for m in ready] + (["Comparar ambos"] if len(ready) == 2 else []),
                          horizontal=True)
        chosen = [m for m in ready if NAMES[m] == choice] or list(ready)
        source = st.radio("Fuente del estudio", ["Subir archivos DICOM", "Estudio de PRUEBA en caché"],
                          horizontal=True)
        vol, true_label = None, None
        if source == "Subir archivos DICOM":
            st.caption("Sube todos los .dcm de UN estudio axial de TC cervical (o al menos 24 cortes). "
                       "Se aplica el mismo muestreo y preprocesamiento que en entrenamiento.")
            files = st.file_uploader("Archivos .dcm", type=["dcm"], accept_multiple_files=True)
            if files:
                vol, meta, bad = preprocess_uploaded(files)
                if bad:
                    st.warning(f"{len(bad)} archivo(s) no legibles: " + "; ".join(bad[:5]))
                if meta is not None:
                    with st.expander(f"Metadatos de los {len(meta)} cortes usados"):
                        st.dataframe(meta)
                    if meta["is_hu"].eq(False).any():
                        st.warning("Algunos cortes no tienen RescaleSlope/Intercept: la escala HU no está garantizada.")
        else:
            sp = pd.read_csv(RESULTS_DIR / "splits.csv")
            test = sp[(sp.split == "test") & sp.StudyInstanceUID.map(lambda u: (CACHE_DIR / f"{u}.npy").exists())]
            if len(test):
                uid = st.selectbox("Estudio del conjunto de prueba (no visto en entrenamiento)", test.StudyInstanceUID)
                vol = np.load(CACHE_DIR / f"{uid}.npy")
                true_label = int(test.set_index("StudyInstanceUID").loc[uid, "patient_overall"])
            else:
                st.info("No hay volúmenes de prueba en caché local.")
        if vol is not None:
            results = {}
            for m in chosen:
                t = time.time()
                p, ps = predict_volume(m, vol)
                results[m] = dict(p=p, ps=ps, ms=(time.time() - t) * 1000,
                                  thr=ready[m]["metrics"]["test_thr_val_youden"]["threshold"])
            cols = st.columns(len(results))
            for c, (m, r) in zip(cols, results.items()):
                label = "fractura (≥ umbral)" if r["p"] >= r["thr"] else "sin fractura (< umbral)"
                c.plotly_chart(gauge(r["p"], r["thr"], NAMES[m]), use_container_width=True)
                c.markdown(f"**Predicción del modelo:** {label}  \nUmbral (elegido en validación): "
                           f"{r['thr']:.3f} · inferencia {r['ms']:.0f} ms")
            if true_label is not None:
                st.info(f"Etiqueta real del dataset (NO es una predicción): patient_overall = {true_label}")
            mil_ps = results.get("mil", {}).get("ps")
            show_volume(vol, key="pred_slider", slice_probs=mil_ps if interactive else None)

# 4 · Métricas -----------------------------------------------------------------
with tabs[3]:
    if not models:
        st.info("Métricas no disponibles: aún no se ha entrenado ni evaluado ningún modelo.")
    for m, v in models.items():
        mt = v["metrics"]
        st.subheader(NAMES[m])
        t = mt["test_thr_val_youden"]
        cs = st.columns(6)
        for c, k in zip(cs, ["roc_auc", "sensitivity", "specificity", "precision", "f1", "accuracy"]):
            c.metric(k, f"{t[k]:.3f}")
        lo, hi = mt["test_auc_ci95_bootstrap"]
        st.caption(f"Prueba n={t['n']} · AUC IC95 % bootstrap [{lo:.3f}, {hi:.3f}] · umbral {t['threshold']:.3f} "
                   f"(validación) · mejor época {mt['best_epoch']} · entrenamiento {mt['train_time_s']:.0f} s")
        if interactive:
            cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
            a, b = st.columns(2)
            a.plotly_chart(px.imshow(cm, text_auto=True, x=["pred 0", "pred 1"], y=["real 0", "real 1"],
                                     color_continuous_scale="Blues", title="Matriz de confusión (prueba)"),
                           use_container_width=True)
            h = pd.read_csv(RESULTS_DIR / f"history_{m}.csv")
            b.plotly_chart(px.line(h, x="epoch", y=["train_loss", "val_loss", "val_auc"], markers=True,
                                   title="Curvas de entrenamiento"), use_container_width=True)
        else:
            a, b, c = st.columns(3)
            for col, kind in zip((a, b, c), ("confusion", "roc", "curvas")):
                if (FIG_DIR / f"{kind}_{m}.png").exists():
                    col.image(str(FIG_DIR / f"{kind}_{m}.png"))
        with st.expander("JSON completo"):
            st.json(mt)

# 5 · Comparación --------------------------------------------------------------
with tabs[4]:
    if len(models) < 2:
        st.info("La comparación requiere ambos modelos evaluados. "
                + ("Solo hay uno disponible." if models else "No hay modelos evaluados."))
    else:
        comp = RESULTS_DIR / "comparacion_modelos.csv"
        if comp.exists():
            st.dataframe(pd.read_csv(comp))
        if interactive:
            fig = go.Figure()
            for m in models:
                p = pd.read_csv(RESULTS_DIR / f"test_predictions_{m}.csv")
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(p.patient_overall, p.prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{NAMES[m]} AUC={models[m]['metrics']['test_thr_val_youden']['roc_auc']:.3f}"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                     showlegend=False))
            fig.update_layout(title="ROC en prueba", xaxis_title="1 - especificidad", yaxis_title="sensibilidad")
            st.plotly_chart(fig, use_container_width=True)
            rows = [dict(modelo=NAMES[m], métrica=k, valor=v["metrics"]["test_thr_val_youden"][k])
                    for m, v in models.items() for k in ["sensitivity", "specificity", "precision", "f1", "roc_auc"]]
            st.plotly_chart(px.bar(pd.DataFrame(rows), x="métrica", y="valor", color="modelo", barmode="group",
                                   range_y=[0, 1], title="Métricas en prueba"), use_container_width=True)
        else:
            for f in ["comparacion_roc_prueba.png", "comparacion_metricas_prueba.png", "comparacion_tiempos.png"]:
                if (FIG_DIR / f).exists():
                    st.image(str(FIG_DIR / f))
        rep = RESULTS_DIR / "INFORME_RESULTADOS.md"
        if rep.exists():
            with st.expander("Informe de resultados generado"):
                st.markdown(rep.read_text(encoding="utf-8"))

# 6 · Limitaciones -------------------------------------------------------------
with tabs[5]:
    st.markdown(LIMITS)
