import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import MODELS, Config
from .data import build_index, make_split
from .models import norm_stats
from .train import get_device, train_model


def choose_seeds(a, sim: pd.DataFrame | None) -> dict:
    """Seed de cada modelo: --seed-cnn/--seed-vit > --seed > --best (maior F1 na simulação)."""
    seeds = {}
    for key in a.models:
        seed = getattr(a, f"seed_{key}")
        if seed is None:
            seed = a.seed
        if seed is None:
            if sim is None:
                raise SystemExit("informe --seed ou rode as simulações (simulations.csv) para usar --best")
            rows = sim[sim.model == key]
            seed = int(rows.loc[rows.f1_macro.idxmax(), "seed"])
        seeds[key] = seed
    return seeds


def main():
    ap = argparse.ArgumentParser(
        description="Retreina CNN/ViT em um seed das simulações e salva results/models/{cnn,vit}.pt. "
                    "Sem --seed*, usa o seed de maior F1 de cada modelo no simulations.csv.")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=["vit", "cnn"], choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=None, help="mesmo seed para todos os modelos")
    ap.add_argument("--seed-cnn", type=int, default=None, help="seed só da CNN")
    ap.add_argument("--seed-vit", type=int, default=None, help="seed só do ViT")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="só para smoke test offline")
    a = ap.parse_args()

    cfg = Config(results_dir=a.results_dir, data_dir=a.data_dir, epochs=a.epochs,
                 num_workers=a.num_workers, pretrained=not a.no_pretrained)
    sim_csv = cfg.results_dir / "simulations.csv"
    sim = pd.read_csv(sim_csv).drop_duplicates(["seed", "model"], keep="last") if sim_csv.exists() else None
    seeds = choose_seeds(a, sim)
    out_dir = cfg.results_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_index(cfg)

    for key, seed in seeds.items():
        orig = None if sim is None else sim[(sim.model == key) & (sim.seed == seed)]
        if orig is not None and len(orig):
            print(f"[export] {key}: seed {seed} (F1 na simulação = {orig.f1_macro.iloc[0]:.4f}) — treinando...")
        else:
            print(f"[export] {key}: seed {seed} — treinando...")
        split = make_split(df, seed, cfg.val_frac, cfg.test_frac)
        model, row, _, _ = train_model(key, split, cfg, seed, get_device(), verbose=True)
        mean, std = norm_stats(model)
        path = out_dir / f"{key}.pt"
        torch.save(dict(
            key=key, timm_name=MODELS[key], state_dict=model.state_dict(),
            mean=list(mean), std=list(std), img_size=cfg.img_size,
            cache_size=cfg.cache_size, seed=seed,
            test_md5=split["test"]["md5"].tolist(),
            train_md5=split["train"]["md5"].tolist() + split["val"]["md5"].tolist(),
            test_metrics={k: row[k] for k in ("f1_macro", "accuracy", "auc")},
        ), path)
        print(f"[export] {key}: f1={row['f1_macro']:.4f} acc={row['accuracy']:.4f} "
              f"auc={row['auc']:.4f} -> {path}")


if __name__ == "__main__":
    main()
