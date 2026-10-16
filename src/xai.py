"""Explicabilidade (xAI): Grad-CAM nos dois modelos + métrica de fidelidade (deletion).

Retreina CNN e ViT no seed representativo (ΔF1 mais próximo da mediana, escolhido
pelo stats.py) — com seeds fixos o resultado fica próximo ao da simulação original.

Deletion: remove (substitui pela média) os pixels em ordem decrescente de
relevância e mede a queda da probabilidade da classe predita. AUC menor =
explicação mais fiel. Compara-se com uma ordem aleatória como linha de base.

    python -m src.xai --results-dir results
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import CLASS_NAMES, MODEL_LABELS, Config
from .latex import write_macros
from .data import ImageDataset, build_index, build_transforms, make_split
from .models import gradcam_target_layer, norm_stats, vit_reshape_transform
from .plotting import INK_2, MODEL_COLORS, save, setup_style
from .train import get_device, train_model

STEPS = np.linspace(0, 1, 11)


def make_cam(model, key):
    kw = {}
    if key == "vit":
        kw["reshape_transform"] = vit_reshape_transform(model)
    return GradCAM(model=model, target_layers=[gradcam_target_layer(model, key)], **kw)


def denorm(x, mean, std):
    img = x.permute(1, 2, 0).cpu().numpy() * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1).astype(np.float32)


@torch.no_grad()
def deletion_curve(model, x, sal, cls, device):
    """x: (3,H,W) normalizado; sal: (H,W). Retorna prob. da classe a cada passo."""
    h, w = sal.shape
    order = np.argsort(-sal.ravel())
    batch = []
    for frac in STEPS:
        mask = np.ones(h * w, dtype=np.float32)
        mask[order[: int(frac * h * w)]] = 0
        m = torch.from_numpy(mask.reshape(1, h, w))
        batch.append(x * m)  # 0 no espaço normalizado = pixel médio do pré-treino
    logits = model(torch.stack(batch).to(device))
    return torch.softmax(logits.float(), 1)[:, cls].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None, help="padrão: xai_seed de stats.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-per-group", type=int, default=2,
                    help="imagens por grupo (Real/IA × acerto/erro) na grade")
    ap.add_argument("--max-deletion", type=int, default=200)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    setup_style()

    cfg = Config(results_dir=a.results_dir, data_dir=a.data_dir, epochs=a.epochs,
                 pretrained=not a.no_pretrained)
    seed = a.seed
    if seed is None:
        stats_file = cfg.results_dir / "stats.json"
        seed = json.loads(stats_file.read_text())["xai_seed"] if stats_file.exists() else 0
    device = get_device()
    df = build_index(cfg)
    split = make_split(df, seed, cfg.val_frac, cfg.test_frac)
    test = split["test"].reset_index(drop=True)
    print(f"[xai] seed={seed} | teste={len(test)} imagens | device={device}")

    models, inputs, preds, cams, rows = {}, {}, {}, {}, {}
    for key in ["cnn", "vit"]:
        model, row, _, (yt, pt) = train_model(key, split, cfg, seed, device, verbose=False)
        model.eval()
        models[key], rows[key] = model, row
        preds[key] = (pt >= 0.5).astype(int)
        mean, std = norm_stats(model)
        _, t_eval = build_transforms(cfg.img_size, mean, std)
        ds = ImageDataset(test, t_eval)
        inputs[key] = (torch.stack([ds[i][0] for i in range(len(ds))]), mean, std)
        cam = make_cam(model, key)
        maps = []
        for i in range(0, len(ds), 16):
            xb = inputs[key][0][i:i + 16].to(device)
            tg = [ClassifierOutputTarget(int(c)) for c in preds[key][i:i + 16]]
            maps.append(cam(input_tensor=xb, targets=tg))
        cams[key] = np.concatenate(maps)
        print(f"[xai] {key}: f1={row['f1_macro']:.4f} (retreinado no seed do xAI)")
    y = test["label"].to_numpy()

    # ---- Grade Grad-CAM: para cada classe, casos em que ambos acertam e em que divergem/erram.
    rng = np.random.default_rng(seed)
    picks = []
    for lab in [0, 1]:
        both_ok = np.where((y == lab) & (preds["cnn"] == lab) & (preds["vit"] == lab))[0]
        some_err = np.where((y == lab) & ((preds["cnn"] != lab) | (preds["vit"] != lab)))[0]
        for pool in (both_ok, some_err):
            if len(pool):
                picks += list(rng.choice(pool, min(a.n_per_group, len(pool)), replace=False))
    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(1.15 * n + 0.5, 3.9), squeeze=False)
    for c, i in enumerate(picks):
        x_c, mean_c, std_c = inputs["cnn"]
        rgb = denorm(x_c[i], mean_c, std_c)
        axes[0, c].imshow(rgb)
        axes[0, c].set_title(f"Verdadeiro: {CLASS_NAMES[y[i]]}", fontsize=7)
        for r, key in enumerate(["cnn", "vit"], 1):
            overlay = show_cam_on_image(rgb, cams[key][i], use_rgb=True)
            axes[r, c].imshow(overlay)
            ok = preds[key][i] == y[i]
            axes[r, c].set_title(f"pred: {CLASS_NAMES[preds[key][i]]} {'✓' if ok else '✗'}",
                                 fontsize=7, color="#0b0b0b" if ok else "#e34948")
    for r, lab in enumerate(["Imagem", MODEL_LABELS["cnn"], MODEL_LABELS["vit"]]):
        axes[r, 0].set_ylabel(lab, fontsize=7.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "fig_gradcam_grid")

    # ---- Deletion (fidelidade): Grad-CAM vs ordem aleatória.
    idx = np.arange(len(test))[: a.max_deletion]
    curves = {}
    for key in ["cnn", "vit"]:
        x_all = inputs[key][0]
        cam_c, rnd_c = [], []
        for i in idx:
            cls = int(preds[key][i])
            cam_c.append(deletion_curve(models[key], x_all[i], cams[key][i], cls, device))
            rnd_c.append(deletion_curve(models[key], x_all[i], rng.random(cams[key][i].shape),
                                        cls, device))
        curves[key] = dict(cam=np.array(cam_c), rnd=np.array(rnd_c))

    def auc(c):
        return np.trapezoid(c, STEPS, axis=1) if hasattr(np, "trapezoid") else np.trapz(c, STEPS, axis=1)

    metrics = dict(seed=seed, n_images=int(len(idx)))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for key in ["cnn", "vit"]:
        a_cam, a_rnd = auc(curves[key]["cam"]), auc(curves[key]["rnd"])
        metrics[key] = dict(f1=rows[key]["f1_macro"],
                            deletion_auc_gradcam=float(a_cam.mean()),
                            deletion_auc_gradcam_std=float(a_cam.std()),
                            deletion_auc_random=float(a_rnd.mean()),
                            deletion_auc_random_std=float(a_rnd.std()))
        ax.plot(STEPS * 100, curves[key]["cam"].mean(0), color=MODEL_COLORS[key], marker="o",
                markersize=3, label=f"{MODEL_LABELS[key]} Grad-CAM (AUC {a_cam.mean():.3f})")
        ax.plot(STEPS * 100, curves[key]["rnd"].mean(0), color=MODEL_COLORS[key], linestyle="--",
                linewidth=1.1, label=f"{MODEL_LABELS[key]} aleatório (AUC {a_rnd.mean():.3f})")
    ax.set_xlabel("% de pixels removidos (mais relevantes primeiro)")
    ax.set_ylabel("Prob. da classe predita")
    ax.set_title("Deletion: fidelidade das explicações")
    ax.legend(frameon=False, fontsize=6.5)
    ax.tick_params(colors=INK_2)
    save(fig, cfg.figures_dir / "fig_deletion_curves")

    (cfg.results_dir / "xai_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_macros(cfg.results_dir / "xai_macros.tex", dict(
        XaiN=metrics["n_images"],
        DelCnnCam=metrics["cnn"]["deletion_auc_gradcam"],
        DelCnnRnd=metrics["cnn"]["deletion_auc_random"],
        DelVitCam=metrics["vit"]["deletion_auc_gradcam"],
        DelVitRnd=metrics["vit"]["deletion_auc_random"]), digits=3)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
