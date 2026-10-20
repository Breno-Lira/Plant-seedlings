import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import build_index, make_split
from .train import get_device, train_model


def append_csv(path: Path, rows: list[dict]):
    """Acrescenta linhas ao fim do CSV (escreve o cabeçalho só se o arquivo ainda não existe)."""
    df =pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def parse_args():
    """Lê as opções da linha de comando (pasta de resultados, nº de simulações, épocas etc.)."""
    ap =argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="pasta local do dataset (padrão: baixa via kagglehub)")
    ap.add_argument("--n-sim", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--models", nargs="+", default=["cnn", "vit"], choices=["cnn", "vit"])
    ap.add_argument("--no-pretrained", action="store_true", help="só para smoke test offline")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main():
    """Para cada seed, cria o split e treina/avalia CNN e ViT nele, salvando métricas,
    histórico e probabilidades; pula os pares (seed, modelo) já concluídos."""
    a = parse_args()
    cfg = Config(results_dir=a.results_dir, data_dir=a.data_dir, n_sim=a.n_sim,
                 epochs=a.epochs, batch_size=a.batch_size, num_workers=a.num_workers,
                 pretrained=not a.no_pretrained, models=a.models)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = cfg.results_dir / "preds"
    preds_dir.mkdir(exist_ok=True)
    sim_csv = cfg.results_dir / "simulations.csv"
    hist_csv = cfg.results_dir / "history.csv"

    df = build_index(cfg)
    print(f"[run] {len(df)} imagens | Real={int((df.label == 0).sum())} "
          f"IA={int((df.label == 1).sum())} | device={get_device()}")

    done = set()
    if sim_csv.exists():
        prev = pd.read_csv(sim_csv)
        done = set(zip(prev["seed"], prev["model"]))

    for seed in range(cfg.n_sim):
        split = make_split(df, seed, cfg.val_frac, cfg.test_frac)
        for key in cfg.models:
            if (seed, key) in done:
                continue
            t0 = time.perf_counter()
            model, row, hist, (yt, pt) = train_model(key, split, cfg, seed,
                                                     verbose=not a.quiet)
            np.savez(preds_dir / f"seed{seed:02d}_{key}.npz", y=yt, p=pt,
                     md5=split["test"]["md5"].to_numpy())
            
            append_csv(hist_csv, hist)
            append_csv(sim_csv, [row])
            print(f"[run] seed={seed:02d} {key}: f1={row['f1_macro']:.4f} "
                  f"acc={row['accuracy']:.4f} auc={row['auc']:.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[run] concluído -> {sim_csv}")


if __name__ == "__main__":
    main()
