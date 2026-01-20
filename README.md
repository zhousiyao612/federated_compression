# Minimal Federated Compression Framework

This repository provides a **minimal** federated learning setup that supports common model compression strategies, non-IID data splits, CNN/ViT backbones, and transmitted-bit tracking.

## Features

- Compression: Top-K sparsification, quantization, and error-feedback wrappers.
- Non-IID client splits via Dirichlet partitioning.
- Models: `resnet18`, `densenet50`, `vit_small_patch16_224`, `deit_small_patch16_224`.
- Datasets: CIFAR-10/100, MNIST, CelebA.
- Adjustable number of clients.
- Tracks total transmitted bits per training run.

## Installation

```bash
pip install torch torchvision timm
```

## Run Example

```bash
python run_fl.py \
  --dataset cifar10 \
  --model resnet18 \
  --num-clients 10 \
  --non-iid-alpha 0.5 \
  --compression topk \
  --topk-ratio 0.1 \
  --error-feedback
```

### ViT Example

```bash
python run_fl.py \
  --dataset cifar100 \
  --model vit_small_patch16_224 \
  --num-clients 5 \
  --compression quant \
  --quant-bits 8
```

### CelebA Example

```bash
python run_fl.py \
  --dataset celeba \
  --model deit_small_patch16_224 \
  --celeba-attr Smiling
```

## Code Map

- `fl_framework/compression.py`: compression methods + transmitted-bit accounting.
- `fl_framework/data.py`: dataset loaders + non-IID partitioning.
- `fl_framework/models.py`: CNN/ViT model registry.
- `fl_framework/federated.py`: training loop with aggregation and metrics.
- `run_fl.py`: CLI entry point.
