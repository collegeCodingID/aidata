from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import Dataset


class AIDATADataset(Dataset):
    """PyTorch Dataset for sample-level access.

    Use this when individual samples are required (e.g. with
    ``torch.utils.data.DataLoader`` and custom samplers).

    Parameters
    ----------
    path : str
        Path to the AIDATA file.
    cache_size : int
        Chunk cache size for the internal reader.
    return_tensors : bool
        If ``True`` return ``torch.Tensor`` objects,
        otherwise return NumPy arrays.
    """

    def __init__(
        self,
        path: str,
        cache_size: int = 8,
        return_tensors: bool = True,
    ):
        from .reader import AIDATAReader

        self.path = path
        self.return_tensors = return_tensors

        self.reader = AIDATAReader(path, cache_size=cache_size)
        self._length = self._find_length()

    def _find_length(self) -> int:
        for name in ("n_samples", "num_samples", "samples", "length"):
            if hasattr(self.reader, name):
                value = getattr(self.reader, name)
                if isinstance(value, int):
                    return value

        if hasattr(self.reader, "metadata"):
            metadata = self.reader.metadata
            if isinstance(metadata, dict):
                for key in ("n_samples", "num_samples", "samples", "length"):
                    if key in metadata:
                        return int(metadata[key])

        raise RuntimeError("Could not determine number of samples.")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("AIDATA sample index out of range")

        X, y = self.reader.get_batch(start=index, batch_size=1)

        X = np.asarray(X)[0]
        y = np.asarray(y)[0]

        if self.return_tensors:
            X = torch.from_numpy(X)
            y = torch.as_tensor(y)

        return X, y

    def get_batch(self, start: int, batch_size: int):
        """Read a contiguous batch directly.

        Parameters
        ----------
        start : int
            Starting sample index.
        batch_size : int
            Number of samples to read.

        Returns
        -------
        tuple
            ``(X, y)`` as tensors or arrays depending on
            ``return_tensors``.
        """
        X, y = self.reader.get_batch(start=start, batch_size=batch_size)

        X = np.asarray(X)
        y = np.asarray(y)

        if self.return_tensors:
            X = torch.from_numpy(X)
            y = torch.from_numpy(y)

        return X, y

    def cache_info(self):
        """Return cache statistics from the internal reader."""
        if hasattr(self.reader, "cache_info"):
            return self.reader.cache_info()
        return {}

    def clear_cache(self):
        """Clear the internal reader cache."""
        if hasattr(self.reader, "clear_cache"):
            self.reader.clear_cache()

    def close(self):
        """Close the internal reader."""
        if hasattr(self.reader, "close"):
            self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class AIDATABatchDataset(Dataset):
    """PyTorch Dataset that exposes AIDATA chunks/batches directly.

    Each ``__getitem__`` call returns a full batch, avoiding
    per-sample reader overhead.  Best used with
    ``torch.utils.data.DataLoader(..., batch_size=None)``.

    Parameters
    ----------
    path : str
        Path to the AIDATA file.
    batch_size : int
        Number of samples in each batch.
    cache_size : int
        Chunk cache size for the internal reader.
    return_tensors : bool
        If ``True`` return ``torch.Tensor`` objects.
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 256,
        cache_size: int = 8,
        return_tensors: bool = True,
    ):
        self.dataset = AIDATADataset(
            path=path,
            cache_size=cache_size,
            return_tensors=return_tensors,
        )
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.dataset) / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_index):
        if batch_index < 0:
            batch_index += len(self)

        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("AIDATA batch index out of range")

        start = batch_index * self.batch_size
        remaining = len(self.dataset) - start
        size = min(self.batch_size, remaining)

        return self.dataset.get_batch(start=start, batch_size=size)

    def cache_info(self):
        return self.dataset.cache_info()

    def clear_cache(self):
        self.dataset.clear_cache()

    def close(self):
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
