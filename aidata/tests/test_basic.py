import os
import tempfile

import numpy as np
import torch

from aidata import (
    AIDATAWriter,
    AIDATAReader,
    AIDATADataset,
    AIDATABatchDataset,
    AIDATALoader,
)


def test_write_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.aidata")

        X = np.random.rand(1000, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=1000, dtype=np.int64)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=256, verbose=False)

        reader = AIDATAReader(path, cache_size=4)

        assert len(reader) == 1000
        assert reader.metadata["features"] == 10

        X0, y0 = reader[0]
        assert X0.shape == (10,)
        assert y0.shape == ()

        Xb, yb = reader.get_batch(0, 128)
        assert Xb.shape == (128, 10)
        assert yb.shape == (128,)

        reader.close()


def test_aidata_dataset_tensors():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.aidata")

        X = np.random.rand(500, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=500, dtype=np.int64)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=100, verbose=False)

        ds = AIDATADataset(path, return_tensors=True)
        assert len(ds) == 500

        X0, y0 = ds[0]
        assert isinstance(X0, torch.Tensor)
        assert isinstance(y0, torch.Tensor)

        ds.close()


def test_aidata_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.aidata")

        X = np.random.rand(256, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=256, dtype=np.int64)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=64, verbose=False)

        loader = AIDATALoader(path, batch_size=32, shuffle=False)
        assert len(loader) == 8

        batches = list(loader)
        assert len(batches) == 8

        for Xb, yb in batches:
            assert Xb.shape[0] == 32
            assert yb.shape[0] == 32

        loader.close()
