import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torchvision import transforms

from .config import CLASS_NAMES, MODEL_LABELS
from .data import IMG_EXTS, label_from_path, resize_short_side, to_rgb
from .train import get_device

REAL_NAMES = {"real", "reais", "realart", "foto", "fotos"}
AI_NAMES = {"ia", "ai", "fake", "fakes", "aiart", "aiartdata", "sintetica", "sinteticas", "gerada", "geradas"}


def load_checkpoint(path: Path, device):
    """Carrega um .pt do export_model: recria a arquitetura, aplica os pesos e monta o
    mesmo pré-processamento do treino. Retorna (modelo, transformação, checkpoint)."""
    ck =torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["timm_name"], pretrained=False, num_classes=2)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.CenterCrop(ck["img_size"]), transforms.ToTensor(),
        transforms.Normalize(ck["mean"], ck["std"]),
    ])
    ck["seen"] = set(ck.get("train_md5", []))
    return model, tf, ck


def load_image(path: Path, cache_size: int) -> Image.Image:
    """Abre a imagem e a padroniza como no treino: RGB e menor lado = `cache_size` px."""
    with Image.open(path) as im:
        im.load()
        return resize_short_side(to_rgb(im), cache_size)


def true_label(f: Path, root: Path) -> int | None:
    """Rótulo verdadeiro pelo nome da pasta (Real/ -> 0, IA/ -> 1), usado só para conferir
    o resultado depois da predição; None se a imagem não estiver numa pasta rotulada."""
    rel = f.relative_to(root) if root in f.parents else Path(f.name)
    lab = label_from_path(rel.parent)
    if lab is not None:
        return lab
    for part in (p.lower() for p in rel.parent.parts):
        if part in REAL_NAMES:
            return 0
        if part in AI_NAMES:
            return 1
    return None


def collect(inputs):
    """Lista as imagens a classificar (arquivos soltos e pastas, recursivamente) como
    pares (arquivo, pasta raiz)."""
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += [(f, p) for f in sorted(p.rglob("*"))
                      if f.is_file() and f.suffix.lower() in IMG_EXTS]
        else:
            files.append((p, p.parent))
    return files


def report(df: pd.DataFrame, keys: list[str]):
    """Compara predições com os rótulos das pastas e imprime acurácia, F1 e erros por
    classe de cada modelo e do ensemble."""
    lab =df.dropna(subset=["verdadeiro"])
    if "erro" in lab:
        lab = lab[lab["erro"].isna()]
    if lab.empty:
        print("\n(sem rótulos: organize as imagens em subpastas Real/ e IA/ para medir acurácia)")
        return {}
    to_int = {n: i for i, n in enumerate(CLASS_NAMES)}
    out = {}
    print(f"\n=== Desempenho em {len(lab)} imagens rotuladas "
          f"(Real={int((lab.verdadeiro == 'Real').sum())}, IA={int((lab.verdadeiro == 'IA').sum())}) ===")
    for k in keys:
        if k not in lab:
            continue
        ok = lab.dropna(subset=[k])
        y, pred = ok["verdadeiro"].map(to_int).to_numpy(), ok[k].map(to_int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out[k] = dict(accuracy=accuracy_score(y, pred),
                      f1_macro=f1_score(y, pred, average="macro", zero_division=0))
        print(f"{MODEL_LABELS[k]:>16}: acc={out[k]['accuracy']:.4f} f1={out[k]['f1_macro']:.4f} "
              f"| Real certo={tn} Real->IA={fp} | IA certo={tp} IA->Real={fn}")
    return out


def main():
    """Carrega os modelos, classifica cada imagem e mostra a tabela, o resumo e as métricas."""
    ap = argparse.ArgumentParser(
        description="Classifica imagens (Real vs IA). Se a pasta tiver subpastas Real/ e IA/, "
                    "mede acurácia e F1 de cada modelo.")
    ap.add_argument("inputs", nargs="+", help="imagens e/ou pastas")
    ap.add_argument("--models-dir", type=Path, default=Path("results/models"))
    ap.add_argument("--models", nargs="+", default=["vit", "cnn"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cam", action="store_true", help="salva o Grad-CAM de cada imagem em --out-dir")
    ap.add_argument("--csv", action="store_true", help="salva a tabela em --out-dir/predicoes.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("predictions"),
                    help="pasta das saídas (criada se não existir)")
    a = ap.parse_args()
    if a.cam or a.csv:
        a.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    loaded = {k: load_checkpoint(a.models_dir / f"{k}.pt", device)
              for k in a.models if (a.models_dir / f"{k}.pt").exists()}
    if not loaded:
        raise SystemExit(f"nenhum checkpoint em {a.models_dir} (rode src.export_model)")
    for k, (_, _, ck) in loaded.items():
        print(f"[modelo] {MODEL_LABELS[k]}: seed {ck['seed']} | teste original "
              f"f1={ck['test_metrics']['f1_macro']:.4f}")

    rows = []
    for f, root in collect(a.inputs):
        lab = true_label(f, root)
        row = {"arquivo": str(f.relative_to(root)) if root in f.parents else f.name,
               "verdadeiro": CLASS_NAMES[lab] if lab is not None else None}
        try:
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            probs, seen = [], []
            for key, (model, tf, ck) in loaded.items():
                x = tf(load_image(f, ck["cache_size"])).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.softmax(model(x).float(), 1)[0, 1].item()
                probs.append(p)
                seen.append(md5 in ck["seen"])
                row[f"p_IA_{key}"] = round(p, 4)
                row[key] = CLASS_NAMES[int(p >= a.threshold)]
                if a.cam:
                    save_cam(model, key, x, ck, f, int(p >= a.threshold), a.out_dir)
            if len(probs) > 1:
                pe = float(np.mean(probs))
                row["p_IA_ens"] = round(pe, 4)
                row["ens"] = CLASS_NAMES[int(pe >= a.threshold)]
            row["visto_no_treino"] = any(seen)
        except Exception as e: 
            row["erro"] = f"{type(e).__name__}: {e}"[:80]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[[c for c in df.columns if c != "erro"] + (["erro"] if "erro" in df else [])]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df.to_string(index=False))

    keys = list(loaded) + (["ens"] if len(loaded) > 1 else [])
    summary = {k: df[k].value_counts().to_dict() for k in keys if k in df}
    print("\nContagem de predições:", json.dumps(summary, ensure_ascii=False))

    n_seen = int(df.get("visto_no_treino", pd.Series(dtype=bool)).fillna(False).sum())
    if n_seen:
        print(f"\nAVISO: {n_seen} imagem(ns) foram usadas no treino/validação do modelo; "
              "o desempenho nelas é otimista. Métricas abaixo só com as imagens não vistas.")
        df_eval = df[~df["visto_no_treino"].fillna(False).astype(bool)]
    else:
        df_eval = df
    report(df_eval, keys)
    if a.csv:
        csv_path = a.out_dir / "predicoes.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[csv] {csv_path}")
    if a.cam:
        print(f"[cam] Grad-CAM salvo em {a.out_dir}")


def save_cam(model, key, x, ck, f: Path, cls: int, out_dir: Path):
    """Gera o Grad-CAM da classe predita e salva a imagem com o mapa de calor sobreposto."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from .models import gradcam_target_layer, vit_reshape_transform

    kw = {"reshape_transform": vit_reshape_transform(model)} if key == "vit" else {}
    cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model, key)], **kw)
    mask = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
    rgb = x[0].permute(1, 2, 0).cpu().numpy() * np.array(ck["std"]) + np.array(ck["mean"])
    overlay = show_cam_on_image(np.clip(rgb, 0, 1).astype(np.float32), mask, use_rgb=True)
    out =out_dir / f"{f.stem}_{key}_{CLASS_NAMES[cls]}.png"
    Image.fromarray(overlay).save(out)


if __name__ == "__main__":
    main()
