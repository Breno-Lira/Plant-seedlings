"""Estilo único das figuras (paper em PDF vetorial + PNG para os slides)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica validada (ordem fixa): azul, laranja, verde-água.
MODEL_COLORS = {"cnn": "#2a78d6", "vit": "#eb6834", "ens": "#1baf7a"}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig] {stem}.pdf/.png")
