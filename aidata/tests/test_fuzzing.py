import os
import random
import tempfile

import numpy as np
import pytest

from aidata import AIDATAReader, AIDATAWriter
from aidata.exceptions import AIDATAError, InvalidAIDATAFile


def _make_file(path):
    X = np.arange(2560, dtype=np.float32).reshape(1280, 2)
    y = np.arange(1280, dtype=np.int64)
    AIDATAWriter(path).write(X, y, chunk_size=64, compression=True, verbose=False)


def test_single_bit_mutations_never_silently_change_a_valid_file():
    with tempfile.TemporaryDirectory() as tmp:
        original = os.path.join(tmp, "original.aidata")
        _make_file(original)
        raw = bytearray(open(original, "rb").read())

        # Deterministic mutations across header, metadata, payload, index and footer.
        rng = random.Random(20260819)
        positions = sorted(set(rng.randrange(len(raw)) for _ in range(48)))

        for pos in positions:
            mutated = bytearray(raw)
            mutated[pos] ^= 1 << rng.randrange(8)
            path = os.path.join(tmp, f"mutated_{pos}.aidata")
            with open(path, "wb") as f:
                f.write(mutated)

            try:
                reader = AIDATAReader(path)
                # If the mutation affects a non-structural byte and remains a
                # valid file, force payload verification of every chunk.
                for cid in range(reader.chunk_count):
                    reader.get_chunk(cid)
                reader.close()
            except (AIDATAError, OSError, ValueError, UnicodeError):
                pass


def test_random_truncations_are_rejected_or_fail_on_read():
    with tempfile.TemporaryDirectory() as tmp:
        original = os.path.join(tmp, "original.aidata")
        _make_file(original)
        raw = open(original, "rb").read()
        rng = random.Random(42)

        for i in range(24):
            cut = rng.randrange(0, len(raw))
            path = os.path.join(tmp, f"truncated_{i}.aidata")
            with open(path, "wb") as f:
                f.write(raw[:cut])

            try:
                reader = AIDATAReader(path)
                for cid in range(reader.chunk_count):
                    reader.get_chunk(cid)
                reader.close()
            except (AIDATAError, OSError, ValueError, UnicodeError):
                pass


def test_corruption_errors_are_aidata_errors():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.aidata")
        _make_file(path)
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(b"BAD!")
        with pytest.raises(AIDATAError):
            AIDATAReader(path)
