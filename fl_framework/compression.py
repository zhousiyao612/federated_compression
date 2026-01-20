"""Compression utilities for federated learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


@dataclass
class CompressionResult:
    tensor: torch.Tensor
    num_bits: int


class Compressor:
    def compress(self, tensor: torch.Tensor) -> CompressionResult:
        raise NotImplementedError

    def decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


@dataclass
class CompressionConfig:
    method: str = "none"
    topk_ratio: float = 0.1
    quant_bits: int = 8
    error_feedback: bool = False


class TopKCompressor(Compressor):
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def compress(self, tensor: torch.Tensor) -> CompressionResult:
        flattened = tensor.flatten()
        k = max(1, int(self.ratio * flattened.numel()))
        values, indices = torch.topk(flattened.abs(), k)
        signed_values = flattened[indices]
        num_index_bits = int(torch.ceil(torch.log2(torch.tensor(flattened.numel(), device=flattened.device))).item())
        num_bits = k * (32 + num_index_bits)
        compressed = torch.zeros_like(flattened)
        compressed[indices] = signed_values
        return CompressionResult(compressed.view_as(tensor), num_bits)


class QuantizationCompressor(Compressor):
    def __init__(self, num_bits: int) -> None:
        self.num_bits = num_bits

    def compress(self, tensor: torch.Tensor) -> CompressionResult:
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            return CompressionResult(tensor.clone(), tensor.numel() * self.num_bits)
        q_levels = 2 ** self.num_bits - 1
        scale = (max_val - min_val) / q_levels
        quantized = torch.round((tensor - min_val) / scale) * scale + min_val
        num_bits = tensor.numel() * self.num_bits
        return CompressionResult(quantized, num_bits)


class ErrorFeedbackCompressor(Compressor):
    def __init__(self, base: Compressor) -> None:
        self.base = base
        self.residuals: Dict[str, torch.Tensor] = {}

    def compress(self, tensor: torch.Tensor, name: str | None = None) -> CompressionResult:
        key = name or "tensor"
        residual = self.residuals.get(key, torch.zeros_like(tensor))
        compensated = tensor + residual
        result = self.base.compress(compensated)
        decompressed = self.base.decompress(result.tensor)
        self.residuals[key] = compensated - decompressed
        return CompressionResult(decompressed, result.num_bits)


class PassThroughCompressor(Compressor):
    def compress(self, tensor: torch.Tensor) -> CompressionResult:
        return CompressionResult(tensor, tensor.numel() * 32)


def build_compressor(config: CompressionConfig) -> Compressor:
    method = config.method.lower()
    if method == "topk":
        base = TopKCompressor(config.topk_ratio)
    elif method == "quant":
        base = QuantizationCompressor(config.quant_bits)
    elif method == "none":
        base = PassThroughCompressor()
    else:
        raise ValueError(f"Unknown compression method: {config.method}")
    if config.error_feedback:
        return ErrorFeedbackCompressor(base)
    return base


def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    compressor: Compressor,
) -> Tuple[Dict[str, torch.Tensor], int]:
    compressed: Dict[str, torch.Tensor] = {}
    total_bits = 0
    for name, tensor in state_dict.items():
        if isinstance(compressor, ErrorFeedbackCompressor):
            result = compressor.compress(tensor, name=name)
        else:
            result = compressor.compress(tensor)
        compressed[name] = result.tensor
        total_bits += result.num_bits
    return compressed, total_bits


def average_state_dicts(state_dicts: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    averaged: Dict[str, torch.Tensor] = {}
    state_dicts = list(state_dicts)
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0)
    return averaged
