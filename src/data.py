"""Download, limpeza, deduplicação, cache pré-processado e splits do dataset."""
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
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def label_from_path(path: Path) -> int | None:
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
    w, h = img.size
    s = size / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def build_index(cfg: Config) -> pd.DataFrame:
    """Varre o dataset, descarta arquivos corrompidos/minúsculos, remove duplicatas
    exatas (md5) e salva uma cópia padronizada (RGB, menor lado = cache_size, PNG).

    A padronização remove pistas triviais de formato/resolução original, que
    poderiam deixar o classificador "trapacear" em vez de aprender artefatos visuais.
    """
    if cfg.index_csv.exists():
        return pd.read_csv(cfg.index_csv)

    root = cfg.data_dir or download_dataset()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    rows, n_bad, n_tiny = [], 0, 0
    # Ordenação sensível a maiúsculas em qualquer SO (igual ao Linux/Colab): o split de
    # cada seed depende dessa ordem, então Windows e Colab precisam gerar a mesma.
    for path in sorted(Path(root).rglob("*"), key=lambda p: p.relative_to(root).parts):
        # Sem extensão também entra (há um JPEG assim no dataset); o PIL valida o conteúdo.
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
    # Mesma imagem nas duas classes é ambígua -> remove todas as cópias.
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
    """Split estratificado 70/15/15 dependente apenas do seed (pareado entre modelos)."""
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
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["cached"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def make_loaders(split, cfg: Config, mean, std, seed: int):
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
