"""Minimal federated learning framework with compression support."""

from .compression import CompressionConfig, ErrorFeedbackCompressor, QuantizationCompressor, TopKCompressor
from .data import DatasetConfig, build_dataloaders
from .federated import FederatedTrainer, FederatedConfig
from .models import build_model

__all__ = [
    "CompressionConfig",
    "ErrorFeedbackCompressor",
    "QuantizationCompressor",
    "TopKCompressor",
    "DatasetConfig",
    "build_dataloaders",
    "FederatedTrainer",
    "FederatedConfig",
    "build_model",
]
