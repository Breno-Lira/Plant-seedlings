# Real vs IA: CNN vs Vision Transformer — estudo comparativo com 30 simulações

2ª avaliação: análise comparativa entre duas abordagens que compartilham duas técnicas e
diferem em uma (contrafactual), avaliadas por teste de hipótese sobre 30 simulações.

**Problema:** classificar se uma imagem é real ou gerada por IA
(dataset [cashbowman/ai-generated-images-vs-real-images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)).

| | Técnica 1 (fixa) | Técnica 2 (contrafactual) | Técnica 3 (fixa) |
|---|---|---|---|
| **Cenário A (H0)** | Transfer Learning (ImageNet) | CNN — ResNet-50 (`resnet50.a1_in1k`, 23,5M) | xAI — Grad-CAM |
| **Cenário B (H1)** | Transfer Learning (ImageNet) | Transformer — ViT-S/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`, 21,7M) | xAI — Grad-CAM |

## Protocolo

1. **Pré-processamento** ([src/data.py](src/data.py)): descarta arquivos corrompidos, remove duplicatas
   exatas (md5; imagens idênticas em classes diferentes são descartadas) e padroniza tudo
   (RGB, menor lado = 256 px, PNG) para que formato/resolução originais não virem atalho.
2. **30 simulações pareadas** ([src/run_simulations.py](src/run_simulations.py)): para cada seed
   `s = 0..29`, um split estratificado 70/15/15 usado pelos **dois** modelos. Treino idêntico:
   AdamW (lr 1e-4, wd 0,05), cosine, até 10 épocas com early stopping (paciência 3) em
   F1 de validação, AMP, batch 32, augment leve (RandomResizedCrop 0,8–1, flip).
3. **Teste de hipótese** ([src/stats.py](src/stats.py)) sobre ΔF1 = F1(ViT) − F1(CNN):
   H0: μΔ = 0, H1: μΔ ≠ 0, α = 0,05. Shapiro-Wilk decide entre **t pareado** e **Wilcoxon**;
   reporta IC 95% bootstrap, Cohen's d_z e correlação rank-biserial. Acurácia e AUC como
   métricas secundárias; ensemble CNN+ViT como análise exploratória.
4. **xAI** ([src/xai.py](src/xai.py)): Grad-CAM nos dois modelos e curva de *deletion*
   (fidelidade da explicação vs ordem aleatória).

## Como rodar (Google Colab, GPU T4)

Abra [notebooks/estudo_comparativo.ipynb](notebooks/estudo_comparativo.ipynb) no Colab, configure os
Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` e execute as células em ordem. Os resultados vão para o
Google Drive e a execução é retomável.

Linha de comando equivalente:

```bash
pip install -r requirements.txt
python -m src.eda             --results-dir results
python -m src.run_simulations --results-dir results --n-sim 30 --epochs 10
python -m src.stats           --results-dir results
python -m src.xai             --results-dir results
python -m src.export_paper    --results-dir results   # copia figuras/macros para paper/
```

## Paper

`paper/main.tex` (IEEEtran, coluna dupla) lê os números de `paper/data/*_macros.tex` e as
figuras de `paper/figures/`, gerados por `src.export_paper` — nenhum valor é digitado à mão.
Para compilar, suba a pasta `paper/` no Overleaf (pdfLaTeX + BibTeX).

## Saídas (`results/`)

| Arquivo | Conteúdo |
|---|---|
| `simulations.csv` | 1 linha por (seed, modelo): acurácia, F1, precisão, recall, AUC, matriz de confusão, tempos |
| `history.csv` | loss/F1 de validação por época |
| `preds/seedXX_{cnn,vit}.npz` | rótulos e probabilidades no teste (ROC, ensemble) |
| `stats.json`, `table_results.tex` | testes de hipótese e tabela pronta para o paper |
| `xai_metrics.json` | deletion AUC (Grad-CAM vs aleatório) |
| `figures/*.pdf, *.png` | figuras do paper (PDF) e dos slides (PNG) |

## Estrutura

```
src/          código (config, data, models, train, run_simulations, stats, xai, eda, plotting)
notebooks/    notebook do Colab
paper/        artigo IEEE (LaTeX, IEEEtran, coluna dupla)
slides/       apresentação
results/      saídas das simulações
```
