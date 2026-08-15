from __future__ import annotations

import math
import random

import torch


class AIDATALoader:
    """Native batch loader for AIDATA.

    Unlike ``torch.utils.data.DataLoader``, this loader lets
    AIDATA control batch reading directly, which can be more
    efficient when chunk alignment matters.

    Parameters
    ----------
    path : str
        Path to the AIDATA file.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle batch order each epoch.
    cache_size : int
        Chunk cache size for the internal reader.
    drop_last : bool
        If ``True`` drop the last incomplete batch.
    device : str or torch.device, optional
        Device to move batches to.
    seed : int
        Random seed for shuffling.

    Example
    -------
    >>> loader = AIDATALoader(
    ...     "training.aidata",
    ...     batch_size=256,
    ...     shuffle=True,
    ... )
    >>> for X, y in loader:
    ...     ...
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 256,
        shuffle: bool = True,
        cache_size: int = 8,
        drop_last: bool = False,
        device: str | torch.device | None = None,
        seed: int = 42,
    ):
        from .reader import AIDATAReader

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = (
            torch.device(device) if device is not None else None
        )
        self.seed = seed
        self.epoch = 0

        self.reader = AIDATAReader(path, cache_size=cache_size)
        self.num_samples = self._get_num_samples()
        self.num_batches = self._calculate_batches()

    # ========================================================
    # Dataset size
    # ========================================================

    def _get_num_samples(self):
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

        raise RuntimeError(
            "Could not determine number of samples from AIDATAReader."
        )

    # ========================================================
    # Number of batches
    # ========================================================

    def _calculate_batches(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):
        return self.num_batches

    # ========================================================
    # Iterator
    # ========================================================

    def __iter__(self):
        self.epoch += 1

        # ----------------------------------------------------
        # Create batch indexes
        # ----------------------------------------------------

        batch_indexes = list(range(self.num_batches))

        # ----------------------------------------------------
        # Shuffle batches
        # ----------------------------------------------------

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batch_indexes)

        # ----------------------------------------------------
        # Read batches
        # ----------------------------------------------------

        for batch_index in batch_indexes:
            start = batch_index * self.batch_size
            remaining = self.num_samples - start
            current_batch_size = min(self.batch_size, remaining)

            # ----------------------------------------------
            # Drop incomplete batch
            # ----------------------------------------------

            if self.drop_last and current_batch_size < self.batch_size:
                continue

            # ----------------------------------------------
            # Read directly from AIDATA
            # ----------------------------------------------

            X, y = self.reader.get_batch(
                start=start,
                batch_size=current_batch_size,
            )

            # ----------------------------------------------
            # NumPy → Torch
            # ----------------------------------------------

            X = torch.from_numpy(X)
            y = torch.from_numpy(y)

            # ----------------------------------------------
            # Move to device
            # ----------------------------------------------

            if self.device is not None:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

            yield X, y

    # ========================================================
    # Cache
    # ========================================================

    def cache_info(self):
        if hasattr(self.reader, "cache_info"):
            return self.reader.cache_info()
        return {}

    def clear_cache(self):
        if hasattr(self.reader, "clear_cache"):
            self.reader.clear_cache()

    # ========================================================
    # Close
    # ========================================================

    def close(self):
        if hasattr(self.reader, "close"):
            self.reader.close()

    # ========================================================
    # Context manager
    # ========================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
