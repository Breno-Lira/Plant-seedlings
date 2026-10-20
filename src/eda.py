import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from PIL import Image

from .config import CLASS_NAMES, Config
from .data import build_index
from .plotting import CLASS_COLORS, save, setup_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--data-dir", type=Path, default=None)
    a = ap.parse_args()
    cfg = Config(results_dir=a.results_dir, data_dir=a.data_dir)
    setup_style()
    df = build_index(cfg)
    out = cfg.figures_dir

    fmt = df.groupby(["format", "label"]).size().unstack(fill_value=0)
    fmt = fmt.reindex(columns=[0, 1], fill_value=0)
    fmt = fmt.loc[fmt.sum(axis=1).sort_values(ascending=False).index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = np.arange(len(fmt))
    w = 0.38
    for i, lab in enumerate([0, 1]):
        axes[0].bar(x + (i - 0.5) * w, fmt[lab], w * 0.92, color=CLASS_COLORS[lab],
                    label=CLASS_NAMES[lab])
    axes[0].set_xticks(x, fmt.index)
    axes[0].set_ylabel("Nº de imagens")
    axes[0].set_title("Formato original por classe")
    axes[0].legend(frameon=False)

    for lab in [0, 1]:
        d = df[df.label == lab]
        axes[1].scatter(d.width, d.height, s=8, alpha=0.6, color=CLASS_COLORS[lab],
                        label=CLASS_NAMES[lab], linewidths=0)
    ticks = [200, 500, 1000, 2000, 5000]
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for axis in (axes[1].xaxis, axes[1].yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("Largura (px)")
    axes[1].set_ylabel("Altura (px)")
    axes[1].set_title("Resolução original")
    axes[1].legend(frameon=False, markerscale=2)
    fig.tight_layout()
    save(fig, out / "fig_eda_formats")

    rng = np.random.default_rng(0)
    n = 6
    fig, axes = plt.subplots(2, n, figsize=(7.2, 2.7))
    for r, lab in enumerate([0, 1]):
        sample = df[df.label == lab].sample(n, random_state=int(rng.integers(1e6)))
        for c, (_, row) in enumerate(sample.iterrows()):
            ax = axes[r, c]
            with Image.open(row.cached) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(CLASS_NAMES[lab], rotation=0, ha="right", va="center")
    save(fig, out / "fig_eda_samples")

    summary = dict(
        n_images=len(df),
        per_class={CLASS_NAMES[k]: int(v) for k, v in df.label.value_counts().sort_index().items()},
        formats={str(k): {CLASS_NAMES[c]: int(fmt.loc[k, c]) for c in [0, 1]} for k in fmt.index},
        median_resolution={CLASS_NAMES[k]: [int(g.width.median()), int(g.height.median())]
                           for k, g in df.groupby("label")},
    )
    report = cfg.results_dir / "data_report.json"
    if report.exists():
        summary["cleaning"] = json.loads(report.read_text())
    (cfg.results_dir / "eda.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
