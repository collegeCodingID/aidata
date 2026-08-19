import os
import struct
import tempfile

import numpy as np
import pytest

from aidata import AIDATAReader, AIDATAWriter
from aidata.exceptions import AIDATAError, InvalidAIDATAFile
from aidata.format import HEADER_FORMAT, HEADER_SIZE


def test_reserved_metadata_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AIDATAError):
            AIDATAWriter(os.path.join(tmp, "x.aidata")).write(
                np.zeros((4, 2), dtype=np.float32),
                np.zeros(4, dtype=np.int64),
                metadata={"samples": 999},
                verbose=False,
            )


def test_object_dtype_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AIDATAError):
            AIDATAWriter(os.path.join(tmp, "x.aidata")).write(
                np.array([[object()]], dtype=object),
                np.zeros(1, dtype=np.int64),
                verbose=False,
            )


def test_checksum_detects_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.aidata")
        AIDATAWriter(path).write(
            np.arange(32, dtype=np.float32).reshape(16, 2),
            np.arange(16, dtype=np.int64),
            chunk_size=8,
            verbose=False,
        )
        with open(path, "r+b") as f:
            # The first payload begins after header + metadata.
            f.seek(0)
            header = f.read(HEADER_SIZE)
            _, _, metadata_size, _ = struct.unpack(HEADER_FORMAT, header)
            f.seek(HEADER_SIZE + metadata_size)
            f.seek(1, 1)
            value = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([value[0] ^ 0xFF]))
        with AIDATAReader(path) as reader:
            with pytest.raises(InvalidAIDATAFile, match="checksum|decompress"):
                reader.get_chunk(0)


def _append_to_first_compressed_x_payload(path):
    """Append one byte to the first compressed X payload and repair the index."""
    from aidata.format import FOOTER_FORMAT, FOOTER_SIZE
    import json

    with open(path, "rb") as f:
        raw = bytearray(f.read())

    index_offset, index_size = struct.unpack(FOOTER_FORMAT, raw[-FOOTER_SIZE:])
    index_container = json.loads(bytes(raw[index_offset:index_offset + index_size]).decode("utf-8"))
    chunks = index_container["chunks"]
    original_insert_at = int(chunks[0]["x_offset"]) + int(chunks[0]["x_size"])

    # Insert a byte at the end of the first X payload. Every subsequent byte
    # shifts by one, including the index and footer.
    body = raw[:index_offset]
    body = body[:original_insert_at] + b"X" + body[original_insert_at:]

    for i, chunk in enumerate(chunks):
        if i == 0:
            chunk["x_size"] = int(chunk["x_size"]) + 1
        chunk["x_offset"] = int(chunk["x_offset"]) + 1 if int(chunk["x_offset"]) >= original_insert_at else int(chunk["x_offset"])
        chunk["y_offset"] = int(chunk["y_offset"]) + 1 if int(chunk["y_offset"]) >= original_insert_at else int(chunk["y_offset"])

    new_index = json.dumps(index_container, sort_keys=True, separators=(",", ":")).encode("utf-8")
    new_index_offset = index_offset + 1
    new_footer = struct.pack(FOOTER_FORMAT, new_index_offset, len(new_index))

    with open(path, "wb") as f:
        f.write(body)
        f.write(new_index)
        f.write(new_footer)


def test_trailing_zlib_data_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.aidata")
        X = np.arange(64, dtype=np.float32).reshape(32, 2)
        y = np.arange(32, dtype=np.int64)
        AIDATAWriter(path).write(X, y, chunk_size=16, compression=True, verbose=False)
        _append_to_first_compressed_x_payload(path)
        with AIDATAReader(path) as reader:
            with pytest.raises(InvalidAIDATAFile, match="Trailing data|decompress"):
                reader.get_chunk(0)


def test_truncated_payload_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.aidata")
        AIDATAWriter(path).write(
            np.arange(128, dtype=np.float32).reshape(64, 2),
            np.arange(64, dtype=np.int64),
            chunk_size=16,
            compression=True,
            verbose=False,
        )
        with open(path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
        with pytest.raises(InvalidAIDATAFile):
            AIDATAReader(path)


def test_integration_api_aliases_canonical_dataset():
    torch = pytest.importorskip("torch")
    from aidata.dataset import AIDATADataset, AIDATABatchDataset
    from aidata.integrations.pytorch import AIDATAPyTorchDataset, AIDATABatchDataset as IntegrationBatchDataset
    assert AIDATAPyTorchDataset is AIDATADataset
    assert IntegrationBatchDataset is AIDATABatchDataset
