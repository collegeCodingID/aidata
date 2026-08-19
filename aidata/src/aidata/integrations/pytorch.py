"""PyTorch compatibility API.

The canonical implementations live in :mod:`aidata.dataset`. This module
keeps the historical ``aidata.integrations.pytorch`` import path stable.
"""

from ..dataset import AIDATADataset, AIDATABatchDataset

# Backward-compatible name retained for users of the integrations namespace.
AIDATAPyTorchDataset = AIDATADataset

__all__ = ["AIDATAPyTorchDataset", "AIDATABatchDataset"]
