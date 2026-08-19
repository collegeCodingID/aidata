from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np

from .exceptions import AIDATAError
from .format import FOOTER_FORMAT, HEADER_FORMAT, MAGIC, VERSION


class AIDATAWriter:
    """Write NumPy arrays to the AIDATA v1 chunked binary format."""

    RESERVED_METADATA_KEYS = {
        "version", "samples", "features", "x_shape", "y_shape",
        "x_dtype", "y_dtype", "x_ndim", "y_ndim", "compression",
        "chunk_size", "compression_level", "checksum", "metadata_sha256", "format",
    }

    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def _canonical_metadata_bytes(metadata: dict) -> bytes:
        return json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def write(
        self,
        X,
        y,
        metadata=None,
        compression=True,
        chunk_size=4096,
        compression_level=3,
        verbose=True,
    ):
        X = np.asarray(X)
        y = np.asarray(y)

        if not isinstance(compression, (bool, np.bool_)):
            raise AIDATAError("compression must be a boolean")
        if X.ndim < 2:
            raise AIDATAError(f"X must be at least 2D. Got shape {X.shape}")
        if y.ndim < 1:
            raise AIDATAError(f"y must be at least 1D. Got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise AIDATAError(
                f"X and y must contain the same samples. Got X={len(X)}, y={len(y)}"
            )
        if X.shape[0] > 0 and (any(int(v) == 0 for v in X.shape[1:]) or any(int(v) == 0 for v in y.shape[1:])):
            raise AIDATAError("X and y dimensions after the sample axis must be non-zero")
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise AIDATAError("chunk_size must be a positive integer")
        if not isinstance(compression_level, (int, np.integer)):
            raise AIDATAError("compression_level must be an integer")
        if compression and not 1 <= compression_level <= 9:
            raise AIDATAError("compression_level must be an integer from 1 to 9 when compression is enabled")
        if not compression:
            compression_level = 0
        if X.dtype.hasobject or y.dtype.hasobject:
            raise AIDATAError("object dtype is not supported; use a fixed-width NumPy dtype")
        if X.dtype.fields is not None or y.dtype.fields is not None:
            raise AIDATAError("structured dtypes are not supported")
        if X.dtype.kind == "V" or y.dtype.kind == "V":
            raise AIDATAError("void dtypes are not supported")

        # AIDATA stores native-endian bytes so readers behave consistently across machines.
        if not X.dtype.isnative:
            X = X.astype(X.dtype.newbyteorder("="), copy=False)
        if not y.dtype.isnative:
            y = y.astype(y.dtype.newbyteorder("="), copy=False)

        if metadata is not None and not isinstance(metadata, dict):
            raise AIDATAError("metadata must be a dictionary or None")
        user_metadata = {} if metadata is None else dict(metadata)
        reserved = self.RESERVED_METADATA_KEYS.intersection(user_metadata)
        if reserved:
            raise AIDATAError(f"metadata contains reserved keys: {sorted(reserved)}")

        compression_name = "zlib" if compression else "none"
        file_metadata = {
            "format": "AIDATA",
            "version": VERSION,
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]) if X.ndim == 2 else list(X.shape[1:]),
            "x_shape": list(X.shape),
            "y_shape": list(y.shape),
            "x_dtype": X.dtype.str,
            "y_dtype": y.dtype.str,
            "x_ndim": int(X.ndim),
            "y_ndim": int(y.ndim),
            "compression": compression_name,
            "compression_level": int(compression_level) if compression else 0,
            "chunk_size": int(chunk_size),
            "checksum": "crc32",
            **user_metadata,
        }

        # Protect metadata itself against accidental corruption/tampering.
        try:
            file_metadata["metadata_sha256"] = hashlib.sha256(
                self._canonical_metadata_bytes(file_metadata)
            ).hexdigest()
            metadata_bytes = self._canonical_metadata_bytes(file_metadata)
        except (TypeError, ValueError) as exc:
            raise AIDATAError("metadata must contain only JSON-serializable values") from exc
        if len(metadata_bytes) > 0xFFFFFFFF:
            raise AIDATAError("metadata is too large for the AIDATA v1 format")

        compressor = (lambda data: zlib.compress(data, level=int(compression_level))) if compression else None
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a sibling temporary file, then atomically replace the destination.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        os.close(fd)

        index = []
        chunk_count = 0
        try:
            with open(tmp_name, "wb") as f:
                # chunk_count is patched after all chunks are written.
                f.write(struct.pack(HEADER_FORMAT, MAGIC, VERSION, len(metadata_bytes), 0))
                f.write(metadata_bytes)

                for start in range(0, X.shape[0], int(chunk_size)):
                    end = min(start + int(chunk_size), X.shape[0])
                    x_raw = np.ascontiguousarray(X[start:end]).tobytes(order="C")
                    y_raw = np.ascontiguousarray(y[start:end]).tobytes(order="C")
                    x_data = compressor(x_raw) if compressor else x_raw
                    y_data = compressor(y_raw) if compressor else y_raw

                    x_offset = f.tell()
                    f.write(x_data)
                    y_offset = f.tell()
                    f.write(y_data)

                    index.append({
                        "start": int(start),
                        "end": int(end),
                        "x_offset": int(x_offset),
                        "y_offset": int(y_offset),
                        "x_size": len(x_data),
                        "y_size": len(y_data),
                        "x_raw_size": len(x_raw),
                        "y_raw_size": len(y_raw),
                        "x_crc32": zlib.crc32(x_raw) & 0xFFFFFFFF,
                        "y_crc32": zlib.crc32(y_raw) & 0xFFFFFFFF,
                    })
                    chunk_count += 1

                index_offset = f.tell()
                try:
                    index_bytes = self._canonical_metadata_bytes({"chunks": index})
                except (TypeError, ValueError) as exc:
                    raise AIDATAError("failed to serialize AIDATA index") from exc
                if len(index_bytes) > 0xFFFFFFFF:
                    raise AIDATAError("index is too large for the AIDATA v1 format")
                f.write(index_bytes)
                f.write(struct.pack(FOOTER_FORMAT, index_offset, len(index_bytes)))

                f.seek(0)
                f.write(struct.pack(
                    HEADER_FORMAT, MAGIC, VERSION, len(metadata_bytes), chunk_count
                ))
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

        if verbose:
            print(f"File created : {self.path}")
            print(f"Samples      : {len(X)}")
            print(f"X shape      : {X.shape}")
            print(f"y shape      : {y.shape}")
            print(f"Chunks       : {chunk_count}")
            print(f"Chunk size   : {chunk_size}")
            print(f"Compression  : {compression_name}")
            print("Checksum     : crc32")
