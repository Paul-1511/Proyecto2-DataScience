"""Fases 4-6: entrenamiento y evaluación a nivel de estudio.

Modelos (la etiqueta es SOLO a nivel de estudio: patient_overall):
  mil   : ResNet18 (ImageNet) 2D por corte -> logit por corte -> agregación explícita:
          logit_estudio = media de los TOP_K logits de corte más altos (MIL débil).
          No se asume que todos los cortes de un estudio positivo contienen fractura.
  bigru : ResNet18 (ImageNet) por corte -> secuencia de 512-d ordenada por z ->
          GRU bidireccional -> [media, máx] temporal -> logit de estudio.
Entrada: v1 [K,224,224] (canal replicado ×3) o v2 2.5D [K,3,224,224] (vecinos ±CONTEXT_MM).
Selección de época (early stopping) y umbral: SOLO con validación. La prueba se evalúa una
única vez, al final, con el checkpoint elegido en validación.
Salidas (RESULTS_DIR): metrics_<m>.json, history_<m>.csv, val/test_predictions_<m>.csv, figuras.
Checkpoint (WEIGHTS_DIR): <m>_final.pt = {state_dict, config, threshold, ...} (ver inference.py).
Uso: python s04_train_eval.py --model mil|bigru [--epochs 15 --patience 4 --bs 4 --accum 2]
"""
import argparse
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torchvision.models import ResNet18_Weights, resnet18

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (EXP, FIG_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR, preprocessing_config,  # noqa: E402
                    seed_everything)

TOP_K = 3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class Backbone(nn.Module):
    """ResNet18. pretrained=True: pesos ImageNet de torchvision o, sin internet, del archivo
    indicado en RESNET18_WEIGHTS. pretrained=False: arquitectura vacía (para cargar checkpoints)."""

    def __init__(self, pretrained=True):
        super().__init__()
        local = os.environ.get("RESNET18_WEIGHTS")
        if pretrained and local:
            m = resnet18(weights=None)
            m.load_state_dict(torch.load(local, map_location="cpu"))
        else:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):  # x [B,K,3,H,W] -> [B,K,512]
        b, k = x.shape[:2]
        return self.net(x.flatten(0, 1)).view(b, k, -1)


class MILResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = Backbone(pretrained)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, x):
        slice_logits = self.head(self.backbone(x)).squeeze(-1)        # [B,K]
        study_logit = slice_logits.topk(TOP_K, dim=1).values.mean(1)  # agregación top-k
        return study_logit, slice_logits


class ResNetBiGRU(nn.Module):
    def __init__(self, hidden=128, pretrained=True):
        super().__init__()
        self.backbone = Backbone(pretrained)
        self.gru = nn.GRU(512, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, 1))

    def forward(self, x):
        h, _ = self.gru(self.backbone(x))                              # [B,K,2H]
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(pooled).squeeze(-1), None


def build_model(name, pretrained=True):
    return MILResNet(pretrained) if name == "mil" else ResNetBiGRU(pretrained=pretrained)


class StudyDataset(torch.utils.data.Dataset):
    """Lee cada estudio preprocesado bajo demanda (no carga todo en RAM)."""

    def __init__(self, frame):
        from s03_preprocess import load_cached
        self.uids = frame["StudyInstanceUID"].tolist()
        self.y = frame["patient_overall"].astype(np.float32).tolist()
        self.load = load_cached

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        return torch.from_numpy(self.load(self.uids[i])), torch.tensor(self.y[i])


def available(splits, name):
    from s03_preprocess import cache_path
    s = splits[splits["split"] == name]
    return s[[cache_path(u).exists() for u in s["StudyInstanceUID"]]].reset_index(drop=True)


def to_input(xb, device, train):
    """uint8 [B,K,H,W] (v1) o [B,K,3,H,W] (v2 2.5D) -> float normalizado ImageNet [B,K,3,H,W]."""
    x = xb.to(device, non_blocking=True).float().div_(255)
    if x.dim() == 4:
        x = x.unsqueeze(2)                                             # [B,K,1,H,W]
    if train:
        # mismo aumento para todos los cortes del estudio: flip horizontal + desplazamiento
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        dx, dy = (torch.randint(-12, 13, (2,))).tolist()
        x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    if x.shape[2] == 1:
        x = x.expand(-1, -1, 3, -1, -1)
    return (x - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def predict(model, loader, device, with_slices=False):
    model.eval()
    out, sl = [], []
    for xb, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logit, slog = model(to_input(xb, device, False))
        out.append(torch.sigmoid(logit.float()).cpu())
        if with_slices and slog is not None:
            sl.append(torch.sigmoid(slog.float()).cpu())
    p = torch.cat(out).numpy()
    return (p, torch.cat(sl).numpy()) if with_slices else p


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(threshold=float(thr), n=int(len(y)), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                sensitivity=float(recall_score(y, yhat, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn + fp else float("nan"),
                precision=float(precision_score(y, yhat, zero_division=0)),
                f1=float(f1_score(y, yhat, zero_division=0)),
                accuracy=float((yhat == y).mean()),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"))


def bootstrap_auc(y, p, n=2000, seed=SEED):
    rng, aucs = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mil", "bigru"], required=True)
    ap.add_argument("--epochs", type=int, default=12, help="máximo de épocas")
    ap.add_argument("--patience", type=int, default=0, help="early stopping (0 = desactivado)")
    ap.add_argument("--bs", type=int, default=4, help="estudios por paso")
    ap.add_argument("--accum", type=int, default=2, help="acumulación de gradientes")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    args = ap.parse_args()
    while True:  # ante OOM: reducir batch (manteniendo el batch efectivo) y reintentar
        try:
            return run(args)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if args.bs == 1:
                raise
            args.bs, args.accum = args.bs // 2, args.accum * 2
            print(f"OOM -> reintento con bs={args.bs}, accum={args.accum}", flush=True)


def run(args):
    seed_everything()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = pd.read_csv(RESULTS_DIR / "splits.csv")
    ids = {n: set(splits.loc[splits.split == n, "StudyInstanceUID"]) for n in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), \
        "particiones no disjuntas"
    str_, sva, ste = available(splits, "train"), available(splits, "val"), available(splits, "test")
    dl = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    g = torch.Generator().manual_seed(SEED)
    tr_loader = torch.utils.data.DataLoader(StudyDataset(str_), batch_size=args.bs, shuffle=True,
                                            generator=g, drop_last=False, **dl)
    va_loader = torch.utils.data.DataLoader(StudyDataset(sva), batch_size=args.bs, **dl)
    te_loader = torch.utils.data.DataLoader(StudyDataset(ste), batch_size=args.bs, **dl)
    ytr, yva, yte = (f["patient_overall"].values.astype(int) for f in (str_, sva, ste))
    x0, _ = tr_loader.dataset[0]
    print(f"[{EXP}] train {len(ytr)} ({ytr.sum()}+)  val {len(yva)} ({yva.sum()}+)  "
          f"test {len(yte)} ({yte.sum()}+)  entrada {tuple(x0.shape)} {x0.dtype}  device={device}", flush=True)

    model = build_model(args.model).to(device).to(memory_format=torch.channels_last)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # Desbalance: pos_weight = negativos/positivos calculado SOLO con entrenamiento
    pos_weight = torch.tensor((len(ytr) - ytr.sum()) / ytr.sum(), dtype=torch.float32, device=device)
    best_path = WEIGHTS_DIR / f"{args.model}_best_state.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    hist, best_auc, best_ep, t0 = [], -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        n_steps = len(tr_loader)
        for j, (xb, yb) in enumerate(tr_loader):
            x, y = to_input(xb, device, True), yb.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logit, _ = model(x)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"pérdida no finita en época {ep}, paso {j}")
            scaler.scale(loss / args.accum).backward()
            if (j + 1) % args.accum == 0 or j == n_steps - 1:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        sched.step()
        pva = predict(model, va_loader, device)
        va_loss = float(F.binary_cross_entropy(torch.tensor(pva).clamp(1e-6, 1 - 1e-6),
                                               torch.tensor(yva, dtype=torch.float32)))
        va_auc = float(roc_auc_score(yva, pva))
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=va_loss, val_auc=va_auc,
                         epoch_s=round(time.time() - te0, 1), elapsed_s=round(time.time() - t0, 1),
                         lr_backbone=opt.param_groups[0]["lr"],
                         gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0))
        print(hist[-1], flush=True)
        if va_auc > best_auc:
            best_auc, best_ep = va_auc, ep
            torch.save(model.state_dict(), best_path)
        elif args.patience and ep - best_ep >= args.patience:
            print(f"Early stopping: sin mejora de AUC de validación en {args.patience} épocas "
                  f"(mejor época {best_ep}).", flush=True)
            break
    train_time = time.time() - t0

    # --- evaluación: mejor época por AUC de validación; umbral = Youden en validación ---
    model.load_state_dict(torch.load(best_path, map_location=device))
    pva = predict(model, va_loader, device)
    fpr, tpr, thr = roc_curve(yva, pva)
    thr_val = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    res_te = predict(model, te_loader, device, with_slices=args.model == "mil")
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.time() - t1
    pte, slice_probs = res_te if args.model == "mil" else (res_te, None)
    if slice_probs is not None:  # probabilidades por corte (solo interpretación, no localización)
        np.savez_compressed(RESULTS_DIR / "test_slice_probs_mil.npz", uids=ste["StudyInstanceUID"].values,
                            probs=slice_probs.round(4))

    hparams = dict(epochs_max=args.epochs, patience=args.patience, batch_size=args.bs,
                   grad_accum=args.accum, lr_backbone=args.lr, lr_head=args.lr * 10,
                   weight_decay=1e-4, scheduler="CosineAnnealingLR", optimizer="AdamW",
                   loss="BCEWithLogits(pos_weight)", pos_weight=round(float(pos_weight), 4),
                   amp=device.type == "cuda", augment="flip horizontal + desplazamiento ±12 px",
                   top_k=TOP_K if args.model == "mil" else None,
                   gru_hidden=128 if args.model == "bigru" else None, seed=SEED)
    res = dict(model=args.model, exp=EXP, n_params=sum(p.numel() for p in model.parameters()),
               input_shape=list(x0.shape), best_epoch=best_ep, epochs_run=len(hist),
               train_time_s=round(train_time, 1), test_inference_time_s=round(infer_time, 2),
               inference_ms_per_study=round(1000 * infer_time / len(yte), 1),
               gpu_peak_gb=hist[-1]["gpu_peak_gb"],
               n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)),
               n_train_planned=len(ids["train"]), n_val_planned=len(ids["val"]), n_test_planned=len(ids["test"]),
               device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               torch_version=torch.__version__, hparams=hparams, preprocessing=preprocessing_config(),
               val=metrics(yva, pva, thr_val),
               test_thr_val_youden=metrics(yte, pte, thr_val),
               test_thr_0_5=metrics(yte, pte, 0.5),
               test_auc_ci95_bootstrap=bootstrap_auc(yte, pte),
               aggregation=("media de top-%d logits de corte" % TOP_K if args.model == "mil"
                            else "BiGRU sobre secuencia ordenada por z + [media,max]"))
    (RESULTS_DIR / f"metrics_{args.model}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(hist).to_csv(RESULTS_DIR / f"history_{args.model}.csv", index=False)
    sva.assign(prob=pva, pred=(pva >= thr_val).astype(int)).to_csv(
        RESULTS_DIR / f"val_predictions_{args.model}.csv", index=False)
    ste.assign(prob=pte, pred=(pte >= thr_val).astype(int)).to_csv(
        RESULTS_DIR / f"test_predictions_{args.model}.csv", index=False)
    # Checkpoint autocontenido para el equipo (inference.py)
    torch.save(dict(state_dict=model.state_dict(), model_name=args.model, threshold=thr_val,
                    preprocessing=preprocessing_config(), hparams=hparams, best_epoch=best_ep,
                    val_auc=best_auc, torch_version=torch.__version__,
                    note="Clasificador global por estudio; no localiza fracturas; sin validación clínica."),
               WEIGHTS_DIR / f"{args.model}_final.pt")

    # --- figuras ---
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("BCE loss"); ax[0].legend(); ax[0].set_xlabel("época")
    ax[1].plot(h.epoch, h.val_auc, marker="o"); ax[1].axvline(best_ep, color="k", ls=":", lw=1)
    ax[1].set_title("ROC-AUC validación (línea: época elegida)"); ax[1].set_xlabel("época")
    fig.suptitle(f"Curvas de entrenamiento - {args.model} [{EXP}]"); fig.tight_layout()
    fig.savefig(FIG_DIR / f"curvas_{args.model}.png", dpi=110); plt.close(fig)

    m = res["test_thr_val_youden"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["real 0", "real 1"])
    ax.set_title(f"Prueba - {args.model} (umbral {m['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"confusion_{args.model}.png", dpi=110); plt.close(fig)

    fpr, tpr, _ = roc_curve(yte, pte)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, label=f"AUC={m['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("sensibilidad")
    ax.set_title(f"ROC prueba - {args.model}"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"roc_{args.model}.png", dpi=110); plt.close(fig)

    print(json.dumps({k: res[k] for k in ["train_time_s", "best_epoch", "epochs_run", "test_thr_val_youden",
                                          "test_auc_ci95_bootstrap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
