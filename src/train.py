"""Treino com early stopping, avaliação e métricas de um modelo em um split."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .config import Config
from .data import make_loaders
from .models import count_params, create_model, norm_stats


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, device, amp: bool):
    """Retorna (rótulos, probabilidade da classe IA, loss média, ms por imagem)."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    ys, ps, loss, n = [], [], 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        loss += crit(logits.float(), y).item()
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
        ys.append(y.cpu())
        n += len(y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / max(n, 1)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), loss / max(n, 1), ms


def compute_metrics(y, p, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return dict(
        accuracy=accuracy_score(y, pred),
        f1_macro=f1_score(y, pred, average="macro", zero_division=0),
        precision_macro=precision_score(y, pred, average="macro", zero_division=0),
        recall_macro=recall_score(y, pred, average="macro", zero_division=0),
        auc=auc, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def train_model(key: str, split, cfg: Config, seed: int, device=None, verbose=True):
    """Treina `key` ('cnn' ou 'vit') no split dado. Retorna (modelo, linha de
    resultado, histórico por época, (y_test, p_test))."""
    device = device or get_device()
    set_seed(seed)
    model = create_model(key, pretrained=cfg.pretrained).to(device)
    mean, std = norm_stats(model)
    loaders = make_loaders(split, cfg, mean, std, seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    crit = nn.CrossEntropyLoss()

    best = dict(f1=-1.0, loss=float("inf"), state=None, epoch=0)
    history, bad_epochs = [], 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(y)
            n += len(y)
        sched.step()

        yv, pv, v_loss, _ = predict(model, loaders["val"], device, cfg.amp)
        v_f1 = compute_metrics(yv, pv)["f1_macro"]
        history.append(dict(seed=seed, model=key, epoch=epoch, train_loss=tr_loss / max(n, 1),
                            val_loss=v_loss, val_f1=v_f1))
        if verbose:
            print(f"  [{key} s{seed}] ep{epoch:02d} train_loss={tr_loss / max(n, 1):.4f} "
                  f"val_loss={v_loss:.4f} val_f1={v_f1:.4f}")

        if v_f1 > best["f1"] or (v_f1 == best["f1"] and v_loss < best["loss"]):
            best.update(f1=v_f1, loss=v_loss, epoch=epoch,
                        state=copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    train_time = time.perf_counter() - t0

    model.load_state_dict(best["state"])
    yt, pt, _, ms = predict(model, loaders["test"], device, cfg.amp)
    row = dict(seed=seed, model=key, **compute_metrics(yt, pt),
               best_epoch=best["epoch"], epochs_run=len(history),
               train_time_s=train_time, infer_ms_per_img=ms,
               params=count_params(model), n_train=len(split["train"]),
               n_val=len(split["val"]), n_test=len(split["test"]))
    return model, row, history, (yt, pt)
