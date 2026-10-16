"""Copia figuras (PDF) e macros/tabelas de results/ para paper/, deixando o paper
autocontido (basta subir a pasta paper/ no Overleaf).

    python -m src.export_paper --results-dir results
"""
import argparse
import shutil
from pathlib import Path

TEX_FILES = ["stats_macros.tex", "eda_macros.tex", "xai_macros.tex", "table_results.tex"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--paper-dir", type=Path, default=Path("paper"))
    a = ap.parse_args()
    (a.paper_dir / "figures").mkdir(parents=True, exist_ok=True)
    (a.paper_dir / "data").mkdir(parents=True, exist_ok=True)
    for f in sorted((a.results_dir / "figures").glob("*.pdf")):
        shutil.copy2(f, a.paper_dir / "figures" / f.name)
        print(f"[paper] figures/{f.name}")
    for name in TEX_FILES:
        src = a.results_dir / name
        if src.exists():
            shutil.copy2(src, a.paper_dir / "data" / name)
            print(f"[paper] data/{name}")
        else:
            print(f"[paper] AVISO: {src} não encontrado (rode eda/stats/xai antes)")


if __name__ == "__main__":
    main()
