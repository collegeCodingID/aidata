import inspect

import aidata
from aidata import (
    AIDATAReader,
    AIDATAWriter,
    AIDATADataset,
    AIDATABatchDataset,
    AIDATALoader,
)


def test_public_root_api_is_stable():
    expected = {
        "AIDATAReader",
        "AIDATAWriter",
        "AIDATADataset",
        "AIDATABatchDataset",
        "AIDATALoader",
    }
    assert expected.issubset(set(aidata.__all__))
    for name in expected:
        assert getattr(aidata, name) is not None


def test_legacy_pytorch_import_path_is_compatible():
    from aidata.integrations.pytorch import (
        AIDATAPyTorchDataset,
        AIDATABatchDataset as LegacyBatchDataset,
    )

    assert AIDATAPyTorchDataset is AIDATADataset
    assert LegacyBatchDataset is AIDATABatchDataset


def test_core_constructor_signatures_remain_compatible():
    assert "path" in inspect.signature(AIDATAReader).parameters
    assert "path" in inspect.signature(AIDATAWriter).parameters
    assert "cache_size" in inspect.signature(AIDATAReader).parameters
    assert "batch_size" in inspect.signature(AIDATABatchDataset).parameters
    assert "batch_size" in inspect.signature(AIDATALoader).parameters


def test_version_is_consistent():
    text = open("pyproject.toml", encoding="utf-8").read()
    assert aidata.__version__ == "0.5.6"
    assert 'version = "0.5.6"' in text
