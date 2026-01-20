"""Minimal federated learning entry point."""

from __future__ import annotations

import argparse

from fl_framework import CompressionConfig, DatasetConfig, FederatedConfig, FederatedTrainer, build_dataloaders, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal federated learning with compression")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "mnist", "celeba"])
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--non-iid-alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--compression", default="none", choices=["none", "topk", "quant"])
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--error-feedback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--celeba-attr", default="Smiling")
    parser.add_argument("--device", default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        non_iid_alpha=args.non_iid_alpha,
        celeba_attr=args.celeba_attr,
    )
    client_loaders, test_loader = build_dataloaders(dataset_config)
    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "mnist": 10,
        "celeba": 2,
    }[args.dataset]
    model = build_model(args.model, num_classes=num_classes)
    compression = CompressionConfig(
        method=args.compression,
        topk_ratio=args.topk_ratio,
        quant_bits=args.quant_bits,
        error_feedback=args.error_feedback,
    )
    fed_config = FederatedConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        device=args.device,
        compression=compression,
    )
    trainer = FederatedTrainer(model, client_loaders, test_loader, fed_config)
    _, accuracies = trainer.train()
    for round_id, acc in enumerate(accuracies, start=1):
        print(f"Round {round_id}: accuracy={acc:.4f}")
    print(f"Total transmitted bits: {trainer.total_bits}")


if __name__ == "__main__":
    main()
