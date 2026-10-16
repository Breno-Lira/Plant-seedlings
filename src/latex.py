"""Exporta números dos resultados como macros LaTeX (\\newcommand) para o paper."""
from pathlib import Path


def write_macros(path: Path, values: dict, digits: int = 4):
    lines = []
    for name, v in values.items():
        if isinstance(v, float):
            v = f"{v:.{digits}f}"
        lines.append(rf"\newcommand{{\{name}}}{{{v}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
