import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .config import CLASS_NAMES, MODEL_LABELS
from .plotting import GRID, INK_2, MODEL_COLORS, save, setup_style

ALPHA = 0.05
METRICS = {"f1_macro": "F1-macro", "accuracy": "Acurácia", "auc": "AUC-ROC"}


def rank_biserial(d: np.ndarray) -> float:
    """Tamanho de efeito do Wilcoxon: (W+ - W-) / (W+ + W-), ignorando zeros."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(d: np.ndarray, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """a = CNN (H0), b = ViT (H1). Diferença = b - a."""
    d = b - a
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    out = dict(n=n, mean_cnn=a.mean(), std_cnn=a.std(ddof=1) if n > 1 else float("nan"),
               mean_vit=b.mean(), std_vit=b.std(ddof=1) if n > 1 else float("nan"),
               mean_diff=d.mean(), std_diff=sd, median_diff=float(np.median(d)),
               ci95_diff=bootstrap_ci(d) if n > 1 else [float("nan")] * 2,
               vit_wins=int((d > 0).sum()), cnn_wins=int((d < 0).sum()), ties=int((d == 0).sum()))
    if n >= 3 and sd > 0:
        out["shapiro_W"], out["shapiro_p"] = map(float, stats.shapiro(d))
        t = stats.ttest_rel(b, a)
        out["t_stat"], out["t_p"] = float(t.statistic), float(t.pvalue)
        out["cohen_dz"] = float(d.mean() / sd)
    else:
        out.update(shapiro_W=float("nan"), shapiro_p=float("nan"), t_stat=float("nan"),
                   t_p=float("nan"), cohen_dz=float("nan"))
    if n >= 3 and np.any(d != 0):
        w = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_W"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
    else:
        out["wilcoxon_W"], out["wilcoxon_p"] = float("nan"), 1.0
    out["rank_biserial"] = rank_biserial(d)

    normal = not np.isnan(out["shapiro_p"]) and out["shapiro_p"] > ALPHA
    out["test_used"] = "t pareado" if normal else "Wilcoxon signed-rank"
    out["p_value"] = out["t_p"] if normal else out["wilcoxon_p"]
    out["reject_H0"] = bool(out["p_value"] < ALPHA)
    if out["reject_H0"]:
        out["winner"] = "vit" if d.mean() > 0 else "cnn"
    else:
        out["winner"] = None
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in out.items()}


def load_preds(preds_dir: Path, seeds):
    data = {}
    for s in seeds:
        files = {k: preds_dir / f"seed{s:02d}_{k}.npz" for k in ("cnn", "vit")}
        if all(f.exists() for f in files.values()):
            data[s] = {k: np.load(f, allow_pickle=True) for k, f in files.items()}
    return data


def ensemble_rows(preds) -> pd.DataFrame:
    """Exploratório (fora do teste de hipótese): média das probabilidades CNN+ViT."""
    rows = []
    for s, d in preds.items():
        assert (d["cnn"]["md5"] == d["vit"]["md5"]).all()
        y, p = d["cnn"]["y"], (d["cnn"]["p"] + d["vit"]["p"]) / 2
        pred = (p >= 0.5).astype(int)
        rows.append(dict(seed=s, model="ens", f1_macro=f1_score(y, pred, average="macro"),
                         accuracy=(pred == y).mean(), auc=roc_auc_score(y, p)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- figuras

def fig_distribution(sim, ens, out):
    keys = ["cnn", "vit"] + (["ens"] if len(ens) else [])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    rng = np.random.default_rng(0)
    for ax, (m, name) in zip(axes, METRICS.items()):
        vals = [sim.loc[sim.model == k, m].to_numpy() if k != "ens" else ens[m].to_numpy()
                for k in keys]
        bp = ax.boxplot(vals, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK_2, linewidth=1.4),
                        whiskerprops=dict(color=INK_2, linewidth=0.9),
                        capprops=dict(color=INK_2, linewidth=0.9))
        for patch, k in zip(bp["boxes"], keys):
            patch.set(facecolor=MODEL_COLORS[k] + "33", edgecolor=MODEL_COLORS[k], linewidth=1.2)
        for i, (v, k) in enumerate(zip(vals, keys), 1):
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=10, color=MODEL_COLORS[k],
                       edgecolors="white", linewidths=0.4, zorder=3)
        ax.set_xticks(range(1, len(keys) + 1), [MODEL_LABELS[k].replace(" (", "\n(") for k in keys])
        ax.set_title(name)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Valor no conjunto de teste")
    fig.tight_layout()
    save(fig, out / "fig_boxplot_metrics")


def fig_paired(wide, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for _, r in wide.iterrows():
        up = r.vit > r.cnn
        ax.plot([0, 1], [r.cnn, r.vit], color=MODEL_COLORS["vit" if up else "cnn"],
                alpha=0.45, linewidth=1)
    ax.scatter(np.zeros(len(wide)), wide.cnn, s=12, color=MODEL_COLORS["cnn"], zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(np.ones(len(wide)), wide.vit, s=12, color=MODEL_COLORS["vit"], zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.plot([0, 1], [wide.cnn.mean(), wide.vit.mean()], color="#0b0b0b", linewidth=2.2,
            marker="D", markersize=5, label="Média", zorder=4)
    ax.plot([], [], color=MODEL_COLORS["vit"], alpha=0.6, label="ViT melhor no seed")
    ax.plot([], [], color=MODEL_COLORS["cnn"], alpha=0.6, label="CNN melhor no seed")
    ax.set_xticks([0, 1], [MODEL_LABELS["cnn"], MODEL_LABELS["vit"]])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("F1-macro (teste)")
    ax.set_title("F1 pareado por simulação (linha = 1 seed)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=6.5, handlelength=1.5, columnspacing=1)
    ax.grid(axis="x", visible=False)
    save(fig, out / "fig_paired_f1")


def fig_diff(d, res, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    ax.hist(d, bins=min(12, max(3, len(d) // 2)), color=MODEL_COLORS["vit"] + "99",
            edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK_2, linestyle="--", linewidth=1, label="H0: diferença = 0")
    ax.axvline(d.mean(), color="#0b0b0b", linewidth=1.6, label=f"média = {d.mean():+.4f}")
    lo, hi = res["ci95_diff"]
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, zorder=0, label="IC 95% (bootstrap)")
    ax.set_xlabel("ΔF1 = F1(ViT) − F1(CNN)")
    ax.set_ylabel("Nº de simulações")
    ax.set_title("Distribuição das diferenças pareadas")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    (osm, osr), (slope, inter, _) = stats.probplot(d, dist="norm")
    ax.scatter(osm, osr, s=12, color=MODEL_COLORS["vit"], edgecolors="white", linewidths=0.4)
    ax.plot(osm, slope * np.asarray(osm) + inter, color=INK_2, linewidth=1.2)
    ax.set_xlabel("Quantis teóricos (Normal)")
    ax.set_ylabel("Quantis observados de ΔF1")
    sp = res["shapiro_p"]
    ax.set_title(f"QQ-plot — Shapiro-Wilk p = {sp:.3f}" if not np.isnan(sp) else "QQ-plot")
    fig.tight_layout()
    save(fig, out / "fig_diff_hist_qq")


def fig_confusion(sim, out):
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for ax, k in zip(axes, ["cnn", "vit"]):
        s = sim[sim.model == k][["tn", "fp", "fn", "tp"]].sum()
        cm = np.array([[s.tn, s.fp], [s.fn, s.tp]], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues" if k == "cnn" else "Oranges", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.6 else "#0b0b0b", fontsize=9)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Predito")
        ax.set_title(MODEL_LABELS[k])
        ax.grid(False)
    axes[0].set_ylabel("Verdadeiro")
    fig.tight_layout()
    save(fig, out / "fig_confusion_mean")


def fig_roc(preds, out):
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for k in ["cnn", "vit"]:
        tprs, aucs = [], []
        for d in preds.values():
            y, p = d[k]["y"], d[k]["p"]
            fpr, tpr, _ = roc_curve(y, p)
            tprs.append(np.interp(grid, fpr, tpr))
            tprs[-1][0] = 0
            aucs.append(roc_auc_score(y, p))
        m, s = np.mean(tprs, 0), np.std(tprs, 0)
        ax.plot(grid, m, color=MODEL_COLORS[k],
                label=f"{MODEL_LABELS[k]} (AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
        ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        color=MODEL_COLORS[k], alpha=0.15, linewidth=0)
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title(f"Curva ROC média ± dp ({len(preds)} simulações)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out / "fig_roc_mean")


def fig_history(hist, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for k in ["cnn", "vit"]:
        h = hist[hist.model == k].groupby("epoch")
        for ax, col in zip(axes, ["val_loss", "val_f1"]):
            m, s = h[col].mean(), h[col].std().fillna(0)
            ax.plot(m.index, m, color=MODEL_COLORS[k], label=MODEL_LABELS[k], marker="o",
                    markersize=3)
            ax.fill_between(m.index, m - s, m + s, color=MODEL_COLORS[k], alpha=0.15,
                            linewidth=0)
    axes[0].set_title("Loss de validação (média ± dp)")
    axes[1].set_title("F1-macro de validação (média ± dp)")
    for ax in axes:
        ax.set_xlabel("Época")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out / "fig_training_curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    a = ap.parse_args()
    setup_style()
    rd = a.results_dir
    out = rd / "figures"

    sim = pd.read_csv(rd / "simulations.csv").drop_duplicates(["seed", "model"], keep="last")
    hist = pd.read_csv(rd / "history.csv").drop_duplicates(["seed", "model", "epoch"], keep="last")
    seeds = sorted(set(sim[sim.model == "cnn"].seed) & set(sim[sim.model == "vit"].seed))
    sim = sim[sim.seed.isin(seeds)]
    print(f"[stats] {len(seeds)} simulações pareadas completas")

    results = {}
    for m in METRICS:
        wide = sim.pivot(index="seed", columns="model", values=m).dropna()
        results[m] = paired_test(wide.cnn.to_numpy(), wide.vit.to_numpy())

    preds = load_preds(rd / "preds", seeds)
    ens = ensemble_rows(preds) if preds else pd.DataFrame(columns=["seed", *METRICS])
    if len(ens):
        results["ensemble_exploratory"] = {
            m: dict(mean=float(ens[m].mean()), std=float(ens[m].std(ddof=1)) if len(ens) > 1 else 0.0)
            for m in METRICS}

    # Seed "representativo" para o xAI: ΔF1 mais próximo da mediana.
    wide = sim.pivot(index="seed", columns="model", values="f1_macro")
    d = wide.vit - wide.cnn
    results["xai_seed"] = int((d - d.median()).abs().idxmin())

    r = results["f1_macro"]
    verdict = ("Rejeita H0 -> " + MODEL_LABELS[r["winner"]] + " é superior"
               if r["reject_H0"] else "Não rejeita H0 -> sem diferença significativa")
    results["conclusion"] = verdict
    (rd / "stats.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    fig_distribution(sim, ens, out)
    fig_paired(wide.dropna(), out)
    fig_diff(d.dropna().to_numpy(), r, out)
    fig_confusion(sim, out)
    if preds:
        fig_roc(preds, out)
    fig_history(hist[hist.seed.isin(seeds)], out)

    print(f"\n=== F1-macro: CNN {r['mean_cnn']:.4f}±{r['std_cnn']:.4f} | "
          f"ViT {r['mean_vit']:.4f}±{r['std_vit']:.4f}")
    print(f"ΔF1 = {r['mean_diff']:+.4f}  IC95% {r['ci95_diff']}  Shapiro p={r['shapiro_p']:.4f}")
    print(f"t pareado p={r['t_p']:.4g} (dz={r['cohen_dz']:.3f}) | "
          f"Wilcoxon p={r['wilcoxon_p']:.4g} (r_rb={r['rank_biserial']:.3f})")
    print(f"Teste principal: {r['test_used']} -> {verdict}")


if __name__ == "__main__":
    main()
