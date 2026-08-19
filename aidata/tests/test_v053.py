
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from aidata import AIDATAReader, AIDATAWriter, AIDATADataset
from aidata.exceptions import AIDATAError, InvalidAIDATAFile


def make_file(path, n=64):
    X = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    y = np.arange(n, dtype=np.int64)
    AIDATAWriter(path).write(X, y, chunk_size=8, compression=False, verbose=False)


def test_worker_safe_dataloader_fork():
    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("fork-specific test")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.aidata")
        make_file(path)
        ds = AIDATADataset(path, cache_size=2)
        loader = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
        seen = []
        for xb, yb in loader:
            seen.extend(yb.tolist())
        assert seen == list(range(64))
        ds.close()


def test_dataset_pickle_has_no_open_reader():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.aidata")
        make_file(path)
        ds = AIDATADataset(path)
        ds[0]
        import pickle
        restored = pickle.loads(pickle.dumps(ds))
        assert restored._reader is None
        assert restored[0][1].item() == 0
        ds.close()
        restored.close()


def test_zero_feature_nonempty_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AIDATAError):
            AIDATAWriter(os.path.join(tmp, "x.aidata")).write(
                np.empty((2, 0), dtype=np.float32),
                np.zeros(2, dtype=np.int64),
                compression=False,
                verbose=False,
            )


def test_compression_type_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AIDATAError):
            AIDATAWriter(os.path.join(tmp, "x.aidata")).write(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.int64),
                compression="yes",
                verbose=False,
            )
