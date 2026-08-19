from __future__ import annotations

import math
import os

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .reader import AIDATAReader


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "AIDATA loader requires torch. Install it with: pip install 'aidata[torch]'"
        )


class AIDATALoader:
    """Efficient contiguous-batch training loader.

    ``shuffle=True`` shuffles the order of batches, not individual samples.
    This preserves sequential disk I/O. Use ``shuffle=False`` for deterministic
    sequential access. For true sample-level randomization, use a PyTorch
    DataLoader with ``AIDATADataset`` and an appropriate sampler.
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 256,
        shuffle: bool = True,
        cache_size: int = 8,
        drop_last: bool = False,
        device=None,
        seed: int = 42,
    ):
        _require_torch()
        if not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError("batch_size must be greater than 0 (got {!r})".format(batch_size))
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer (got {!r})".format(cache_size))

        self.path = str(path)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.device = torch.device(device) if device is not None else None
        self.seed = int(seed)
        self.epoch = 0
        self.cache_size = cache_size
        self._reader = None
        self._reader_identity = None
        self._num_samples = None

        self._ensure_reader()
        self.num_samples = len(self._reader)  # type: ignore[arg-type]
        self.num_batches = self._calculate_batches()

    @property
    def reader(self):
        self._ensure_reader()
        return self._reader

    def _ensure_reader(self):
        identity = os.getpid()
        if self._reader is not None and self._reader_identity == identity:
            return
        if self._reader is not None:
            self._reader.close()
        self._reader = AIDATAReader(self.path, cache_size=self.cache_size)
        self._reader_identity = identity

    def _calculate_batches(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self._ensure_reader()
        self.epoch += 1
        batch_indexes = np.arange(self.num_batches, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batch_indexes)

        for batch_index in batch_indexes:
            batch_index = int(batch_index)
            start = batch_index * self.batch_size
            current = min(self.batch_size, self.num_samples - start)
            if self.drop_last and current < self.batch_size:
                continue
            X, y = self.reader.get_batch(start, current)
            X = torch.from_numpy(np.array(X, copy=True))
            y = torch.from_numpy(np.array(y, copy=True))
            if self.device is not None:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            yield X, y

    def cache_info(self):
        return self.reader.cache_info()

    def clear_cache(self):
        self.reader.clear_cache()

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._reader_identity = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
