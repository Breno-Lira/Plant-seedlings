"""Fábrica de modelos (Transfer Learning via timm) e utilitários para Grad-CAM."""
import timm
import torch.nn as nn

from .config import MODELS


def create_model(key: str, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    return timm.create_model(MODELS[key], pretrained=pretrained, num_classes=num_classes)


def norm_stats(model: nn.Module):
    """Média/desvio usados no pré-treino de cada backbone (diferem entre ResNet e ViT)."""
    cfg = timm.data.resolve_data_config({}, model=model)
    return cfg["mean"], cfg["std"]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def gradcam_target_layer(model: nn.Module, key: str) -> nn.Module:
    if key == "cnn":
        return model.layer4[-1]
    return model.blocks[-1].norm1


def vit_reshape_transform(model: nn.Module):
    """Converte a sequência de tokens do ViT (B, 1+N, C) em mapa (B, C, H, W)."""
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    gh, gw = model.patch_embed.grid_size

    def reshape(t):
        t = t[:, n_prefix:, :].reshape(t.size(0), gh, gw, t.size(2))
        return t.permute(0, 3, 1, 2)

    return reshape
