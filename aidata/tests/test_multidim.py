"""Tests for multi-dimensional target support in AIDATA."""

import os
import tempfile

import numpy as np
import torch

from aidata import (
    AIDATAWriter,
    AIDATAReader,
    AIDATADataset,
    AIDATALoader,
)


def test_2d_targets_segmentation():
    """Test segmentation masks: y shape = (N, H, W)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "segmentation.aidata")

        N, H, W = 500, 64, 64
        X = np.random.rand(N, 128).astype(np.float32)  # features
        y = np.random.randint(0, 21, size=(N, H, W), dtype=np.int64)  # 21 classes

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=100, verbose=False)

        reader = AIDATAReader(path, cache_size=4)

        assert len(reader) == N
        assert reader.metadata["y_shape"] == [N, H, W]
        assert reader.metadata["y_ndim"] == 3

        # Single sample
        x0, y0 = reader[0]
        assert x0.shape == (128,)
        assert y0.shape == (H, W)

        # Batch
        Xb, yb = reader.get_batch(0, 50)
        assert Xb.shape == (50, 128)
        assert yb.shape == (50, H, W)

        # Chunk
        Xc, yc = reader.get_chunk(0)
        assert Xc.shape == (100, 128)
        assert yc.shape == (100, H, W)

        reader.close()
        print("✅ 2D targets (segmentation) PASSED")


def test_multilabel_classification():
    """Test multi-label: y shape = (N, num_classes)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "multilabel.aidata")

        N, C = 1000, 20
        X = np.random.rand(N, 512).astype(np.float32)
        y = np.random.randint(0, 2, size=(N, C), dtype=np.int64)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=256, verbose=False)

        reader = AIDATAReader(path)

        x0, y0 = reader[0]
        assert y0.shape == (C,)

        Xb, yb = reader.get_batch(0, 128)
        assert yb.shape == (128, C)

        reader.close()
        print("✅ Multi-label (2D y) PASSED")


def test_3d_targets_volumetric():
    """Test 3D targets: y shape = (N, D, H, W)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "volumetric.aidata")

        N, D, H, W = 100, 16, 32, 32
        X = np.random.rand(N, 256).astype(np.float32)
        y = np.random.rand(N, D, H, W).astype(np.float32)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=32, verbose=False)

        reader = AIDATAReader(path)

        x0, y0 = reader[0]
        assert y0.shape == (D, H, W)

        Xb, yb = reader.get_batch(0, 16)
        assert yb.shape == (16, D, H, W)

        reader.close()
        print("✅ 3D targets (volumetric) PASSED")


def test_pytorch_dataset_multidim():
    """Test AIDATADataset with multi-dimensional y."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "torch_multidim.aidata")

        N, H, W = 256, 32, 32
        X = np.random.rand(N, 64).astype(np.float32)
        y = np.random.randint(0, 10, size=(N, H, W), dtype=np.int64)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=64, verbose=False)

        ds = AIDATADataset(path, return_tensors=True)
        assert len(ds) == N

        x0, y0 = ds[0]
        assert isinstance(x0, torch.Tensor)
        assert isinstance(y0, torch.Tensor)
        assert x0.shape == (64,)
        assert y0.shape == (H, W)

        # Batch via get_batch
        Xb, yb = ds.get_batch(0, 32)
        assert Xb.shape == (32, 64)
        assert yb.shape == (32, H, W)

        ds.close()
        print("✅ PyTorch Dataset multi-dim PASSED")


def test_1d_targets_still_work():
    """Ensure backward compatibility with 1D targets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "classic.aidata")

        X = np.random.rand(1000, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=1000, dtype=np.int64)

        writer = AIDATAWriter(path)
        writer.write(X, y, chunk_size=256, verbose=False)

        reader = AIDATAReader(path)

        assert len(reader) == 1000
        x0, y0 = reader[0]
        assert x0.shape == (20,)
        assert np.isscalar(y0) or y0.shape == ()

        Xb, yb = reader.get_batch(0, 128)
        assert Xb.shape == (128, 20)
        assert yb.shape == (128,)

        reader.close()
        print("✅ 1D targets backward compatibility PASSED")


if __name__ == "__main__":
    test_2d_targets_segmentation()
    test_multilabel_classification()
    test_3d_targets_volumetric()
    test_pytorch_dataset_multidim()
    test_1d_targets_still_work()
    print("\n🎉 All multi-dimensional target tests PASSED!")
