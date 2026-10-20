import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import KAGGLE_DATASET, Config

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".jfif", ".tif", ".tiff"}
MIN_SIDE = 32


def download_dataset() -> Path:
    """Baixa o dataset do Kaggle via kagglehub (sem login) e retorna a pasta local."""
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def label_from_path(path: Path) -> int | None:
    """Rótulo pela pasta: 1 se o caminho contém 'AiArt', 0 se contém 'RealArt', senão None."""
    parts = [p.lower() for p in path.parts]
    if any("aiart" in p for p in parts):
        return 1
    if any("realart" in p for p in parts):
        return 0
    return None


def to_rgb(img: Image.Image) -> Image.Image:
    """Converte qualquer modo (P, LA, RGBA, CMYK, L...) para RGB sobre fundo branco."""
    if img.mode in ("P", "LA", "RGBA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def resize_short_side(img: Image.Image, size: int) -> Image.Image:
    """Redimensiona mantendo a proporção, de modo que o menor lado fique com `size` px."""
    w, h = img.size
    s = size / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def build_index(cfg: Config) -> pd.DataFrame:
    """Monta a tabela de imagens do dataset (salva em index.csv; se já existe, só a lê).

    Descarta arquivos corrompidos ou menores que 32 px, remove duplicatas exatas (md5)
    e grava uma cópia padronizada de cada imagem (RGB, menor lado = 256 px, PNG), para
    que formato e resolução originais não sirvam de pista para o classificador.
    """
    if cfg.index_csv.exists():
        return pd.read_csv(cfg.index_csv)

    root = cfg.data_dir or download_dataset()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    rows, n_bad, n_tiny = [], 0, 0
    
    for path in sorted(Path(root).rglob("*"), key=lambda p: p.relative_to(root).parts):
       
        if (path.suffix and path.suffix.lower() not in IMG_EXTS) or not path.is_file():
            continue
        label = label_from_path(path.relative_to(root))
        if label is None:
            continue
        raw = path.read_bytes()
        try:
            with Image.open(path) as im:
                fmt = im.format
                im.load()
                w, h = im.size
                rgb = to_rgb(im)
        except Exception:
            n_bad += 1
            continue
        if min(w, h) < MIN_SIDE:
            n_tiny += 1
            continue
        md5 = hashlib.md5(raw).hexdigest()
        cached = cfg.cache_dir / f"{md5}.png"
        if not cached.exists():
            resize_short_side(rgb, cfg.cache_size).save(cached)
        rows.append(dict(path=str(path), cached=str(cached), label=label, md5=md5,
                         width=w, height=h, format=fmt, ext=path.suffix.lower(),
                         bytes=len(raw)))

    df = pd.DataFrame(rows)
    n_raw = len(df)
    
    conflict = df.groupby("md5")["label"].nunique()
    df = df[~df["md5"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("md5").reset_index(drop=True)
    report = dict(found=n_raw + n_bad + n_tiny, corrupted=n_bad, tiny=n_tiny,
                  duplicates_removed=n_raw - len(df), kept=len(df))
    df.to_csv(cfg.index_csv, index=False)
    (cfg.results_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"[data] {len(df)} imagens válidas | corrompidas={n_bad} | "
          f"muito pequenas={n_tiny} | duplicadas removidas={n_raw - len(df)}")
    return df


def make_split(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float):
    """Divide em treino/validação/teste (70/15/15), mantendo a proporção Real/IA.

    O resultado depende só do seed, então CNN e ViT recebem a mesma divisão em cada
    simulação. Verifica também que nenhuma imagem aparece em mais de um conjunto.
    """
    idx = np.arange(len(df))
    y = df["label"].to_numpy()
    tr_val, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    rel_val = val_frac / (1 - test_frac)
    train, val = train_test_split(tr_val, test_size=rel_val, stratify=y[tr_val],
                                  random_state=seed)
    split = {"train": df.iloc[train], "val": df.iloc[val], "test": df.iloc[test]}
    hashes = [set(s["md5"]) for s in split.values()]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2]), \
        "vazamento: mesma imagem em mais de um split"
    return split


def build_transforms(img_size: int, mean, std):
    """Retorna as transformações de treino (recorte e espelhamento aleatórios) e de
    avaliação (recorte central), ambas convertendo para tensor normalizado."""
    norm = transforms.Normalize(mean, std)
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    evaluation = transforms.Compose([
        transforms.CenterCrop(img_size), transforms.ToTensor(), norm,
    ])
    return train, evaluation


class ImageDataset(Dataset):
    """Dataset do PyTorch: entrega (imagem transformada, rótulo) a partir das cópias em cache."""

    def __init__(self, df: pd.DataFrame, transform):
        """Guarda os caminhos das imagens em cache, os rótulos e a transformação a aplicar."""
        self.paths = df["cached"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = transform

    def __len__(self):
        """Número de imagens do conjunto."""
        return len(self.paths)

    def __getitem__(self, i):
        """Abre a i-ésima imagem, aplica a transformação e devolve (tensor, rótulo)."""
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


def _seed_worker(worker_id):
    """Fixa o seed do NumPy em cada processo do DataLoader, para resultados reprodutíveis."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def make_loaders(split, cfg: Config, mean, std, seed: int):
    """Cria os DataLoaders de treino (embaralhado, com seed fixo), validação e teste."""
    t_train, t_eval = build_transforms(cfg.img_size, mean, std)
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
              persistent_workers=cfg.num_workers > 0)
    return {
        "train": DataLoader(ImageDataset(split["train"], t_train), shuffle=True,
                            generator=g, drop_last=len(split["train"]) > cfg.batch_size, **kw),
        "val": DataLoader(ImageDataset(split["val"], t_eval), shuffle=False, **kw),
        "test": DataLoader(ImageDataset(split["test"], t_eval), shuffle=False, **kw),
    }
