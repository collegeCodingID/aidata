from __future__ import annotations

import math
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Minimal fallback so the storage layer remains importable without PyTorch."""
        pass

from .reader import AIDATAReader


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch integration requires torch. Install it with: "
            "pip install 'aidata[torch]'"
        )


class AIDATADataset(Dataset):
    """PyTorch-compatible sample-level dataset backed by AIDATA.

    The reader is process/worker-local. This makes the dataset safe to use with
    torch.utils.data.DataLoader(num_workers > 0), including spawn and fork.
    """

    def __init__(self, path: str, cache_size: int = 8, return_tensors: bool = True):
        self.path = str(path)
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError("cache_size must be a positive integer (got {!r})".format(cache_size))
        self.return_tensors = bool(return_tensors)
        self._reader: AIDATAReader | None = None
        self._reader_identity = None

    @property
    def reader(self) -> AIDATAReader:
        self._ensure_reader()
        assert self._reader is not None
        return self._reader

    def _ensure_reader(self) -> None:
        _require_torch()
        worker_id = -1
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
            if info is not None:
                worker_id = int(info.id)
        except Exception:
            pass

        identity = (os.getpid(), worker_id)
        if self._reader is not None and self._reader_identity == identity:
            return

        if self._reader is not None:
            self._reader.close()

        self._reader = AIDATAReader(self.path, cache_size=self.cache_size)
        self._reader_identity = identity

    def __len__(self):
        # Avoid opening a file in the parent process merely to construct a Dataset.
        self._ensure_reader()
        return len(self._reader)  # type: ignore[arg-type]

    def __getitem__(self, index):
        X, y = self.reader[index]
        X = np.array(X, copy=True)
        y = np.array(y, copy=True)
        if self.return_tensors:
            return torch.from_numpy(X), torch.from_numpy(y)
        return X, y

    def get_batch(self, start: int, batch_size: int):
        X, y = self.reader.get_batch(start, batch_size)
        if self.return_tensors:
            return torch.from_numpy(np.array(X, copy=True)), torch.from_numpy(np.array(y, copy=True))
        return X, y

    def cache_info(self):
        return self.reader.cache_info()

    def clear_cache(self):
        self.reader.clear_cache()

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._reader_identity = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # File handles/caches must never be serialized into DataLoader workers.
        state["_reader"] = None
        state["_reader_identity"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __enter__(self):
        self._ensure_reader()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class AIDATABatchDataset(Dataset):
    """Dataset where one item is one contiguous batch."""

    def __init__(self, dataset_or_path, batch_size: int = 256, cache_size: int = 8, return_tensors: bool = True):
        if not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError("batch_size must be greater than 0 (got {!r})".format(batch_size))
        if isinstance(dataset_or_path, AIDATADataset):
            if cache_size != 8 or return_tensors is not True:
                raise ValueError("cache_size/return_tensors cannot be overridden when passing an AIDATADataset")
            self.dataset = dataset_or_path
        else:
            self.dataset = AIDATADataset(
                dataset_or_path, cache_size=cache_size, return_tensors=return_tensors
            )
        self.batch_size = int(batch_size)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, batch_index):
        if not isinstance(batch_index, (int, np.integer)):
            raise TypeError("AIDATA batch index must be an integer")
        batch_index = int(batch_index)
        if batch_index < 0:
            batch_index += len(self)
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("AIDATA batch index out of range")
        return self.dataset.get_batch(batch_index * self.batch_size, self.batch_size)

    def cache_info(self):
        return self.dataset.cache_info()

    def clear_cache(self):
        self.dataset.clear_cache()

    def close(self):
        self.dataset.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
