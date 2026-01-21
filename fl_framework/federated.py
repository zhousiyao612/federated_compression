"""Federated training loop with compression tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .compression import CompressionConfig, average_state_dicts, build_compressor, compress_state_dict


@dataclass
class FederatedConfig:
    rounds: int = 5
    local_epochs: int = 1
    lr: float = 0.01
    device: str = "cpu"
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    log_file: str = "training_log.txt"


class FederatedTrainer:
    def __init__(
        self,
        model: nn.Module,
        client_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        config: FederatedConfig,
    ) -> None:
        self.model = model
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.config = config
        self.compressor = build_compressor(config.compression)
        self.total_bits = 0
        
        # Initialize log file
        os.makedirs(os.path.dirname(self.config.log_file) if os.path.dirname(self.config.log_file) else ".", exist_ok=True)
        with open(self.config.log_file, 'w') as f:
            f.write("Federated Learning Training Log\n")
            f.write("=" * 40 + "\n\n")

    def _train_client(self, model: nn.Module, loader: DataLoader) -> nn.Module:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        for _ in range(self.config.local_epochs):
            for inputs, targets in loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                # Handle batch size of 1 by switching to eval mode for BatchNorm
                if inputs.size(0) == 1:
                    model.eval()
                else:
                    model.train()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                targets = self._normalize_targets(targets)
                loss = F.cross_entropy(outputs, targets.long())
                loss.backward()
                optimizer.step()
        return model

    def _evaluate(self, model: nn.Module) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                outputs = model(inputs)
                targets = self._normalize_targets(targets)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
        return correct / max(total, 1)

    @staticmethod
    def _normalize_targets(targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim > 1:
            targets = targets[:, 0]
        if targets.min() < 0:
            targets = (targets + 1) // 2
        return targets

    def train(self) -> Tuple[nn.Module, Iterable[float]]:
        self.model.to(self.config.device)
        accuracies = []
        for round_num in range(self.config.rounds):
            client_states = []
            for _, loader in self.client_loaders.items():
                client_model = copy.deepcopy(self.model)
                client_model.to(self.config.device)
                trained = self._train_client(client_model, loader)
                compressed_state, num_bits = compress_state_dict(trained.state_dict(), self.compressor)
                self.total_bits += num_bits
                client_states.append(compressed_state)
            averaged = average_state_dicts(client_states)
            self.model.load_state_dict(averaged)
            accuracy = self._evaluate(self.model)
            accuracies.append(accuracy)
            log_message = f"Round {round_num + 1}: accuracy={accuracy:.4f}, total_bits={self.total_bits:,}"
            print(log_message)
            
            # Write to log file
            with open(self.config.log_file, 'a') as f:
                f.write(log_message + "\n")
        return self.model, accuracies
