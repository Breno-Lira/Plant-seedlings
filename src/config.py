"""Hiperparâmetros e caminhos compartilhados por todos os scripts."""
from dataclasses import dataclass, field
from pathlib import Path

KAGGLE_DATASET = "cashbowman/ai-generated-images-vs-real-images"

# Cenário A (H0) = CNN; Cenário B (H1) = Transformer. Mesma família de
# pré-treino (ImageNet) e nº de parâmetros comparável (~25M vs ~22M).
MODELS = {
    "cnn": "resnet50.a1_in1k",
    "vit": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
MODEL_LABELS = {"cnn": "CNN (ResNet-50)", "vit": "ViT-S/16", "ens": "Ensemble"}

CLASS_NAMES = ["Real", "IA"]  # rótulo 0 = Real, 1 = Gerada por IA


@dataclass
class Config:
    results_dir: Path = Path("results")
    data_dir: Path | None = None   # None -> baixa via kagglehub
    cache_size: int = 256          # menor lado das imagens pré-processadas
    img_size: int = 224
    n_sim: int = 30
    epochs: int = 10
    patience: int = 3
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    val_frac: float = 0.15
    test_frac: float = 0.15
    num_workers: int = 2
    pretrained: bool = True
    amp: bool = True
    models: list[str] = field(default_factory=lambda: list(MODELS))

    @property
    def cache_dir(self) -> Path:
        return self.results_dir / "cache"

    @property
    def index_csv(self) -> Path:
        return self.results_dir / "index.csv"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"
