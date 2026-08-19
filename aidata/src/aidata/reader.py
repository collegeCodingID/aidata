from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections import OrderedDict

import numpy as np

from .exceptions import InvalidAIDATAFile, UnsupportedVersion
from .format import FOOTER_FORMAT, FOOTER_SIZE, HEADER_FORMAT, HEADER_SIZE, MAGIC, VERSION


class AIDATAReader:
    """Validated, cached reader for AIDATA v1 files."""

    def __init__(self, path, cache_size=8):
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer (got {!r})".format(cache_size))

        self.path = str(path)
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self._file = None
        self._file_size = 0
        self._data_start = 0
        self._index_offset = 0

        try:
            self._file = open(self.path, "rb")
            self._file.seek(0, 2)
            self._file_size = self._file.tell()
            if self._file_size < HEADER_SIZE + FOOTER_SIZE:
                raise InvalidAIDATAFile("File is too small to be a valid AIDATA file")

            self._file.seek(0)
            header = self._file.read(HEADER_SIZE)
            if len(header) != HEADER_SIZE:
                raise InvalidAIDATAFile("Truncated header")
            magic, version, metadata_size, chunk_count = struct.unpack(HEADER_FORMAT, header)

            if magic != MAGIC:
                raise InvalidAIDATAFile("Invalid AIDATA magic")
            if version != VERSION:
                raise UnsupportedVersion(f"Unsupported AIDATA version: {version}")
            if metadata_size > self._file_size - HEADER_SIZE - FOOTER_SIZE:
                raise InvalidAIDATAFile("Metadata extends beyond file")
            if metadata_size == 0:
                raise InvalidAIDATAFile("Metadata is empty")

            self.version = version
            self.chunk_count = int(chunk_count)
            metadata_bytes = self._file.read(metadata_size)
            if len(metadata_bytes) != metadata_size:
                raise InvalidAIDATAFile("Truncated metadata")
            try:
                self.metadata = json.loads(metadata_bytes.decode("utf-8"))
            except Exception as exc:
                raise InvalidAIDATAFile("Invalid metadata JSON") from exc

            self._validate_metadata(metadata_bytes)
            self._data_start = HEADER_SIZE + metadata_size

            self._file.seek(-FOOTER_SIZE, 2)
            footer = self._file.read(FOOTER_SIZE)
            if len(footer) != FOOTER_SIZE:
                raise InvalidAIDATAFile("Truncated footer")
            index_offset, index_size = struct.unpack(FOOTER_FORMAT, footer)
            self._index_offset = int(index_offset)

            if index_offset < self._data_start:
                raise InvalidAIDATAFile("Invalid index offset")
            if index_offset > self._file_size - FOOTER_SIZE:
                raise InvalidAIDATAFile("Index offset is outside the file")
            if index_size > self._file_size - FOOTER_SIZE - index_offset:
                raise InvalidAIDATAFile("Index extends beyond file")
            if index_offset + index_size != self._file_size - FOOTER_SIZE:
                raise InvalidAIDATAFile("Index does not end immediately before footer")

            self._file.seek(index_offset)
            index_bytes = self._file.read(index_size)
            if len(index_bytes) != index_size:
                raise InvalidAIDATAFile("Truncated index")
            try:
                index_container = json.loads(index_bytes.decode("utf-8"))
                self.index = index_container["chunks"]
            except Exception as exc:
                raise InvalidAIDATAFile("Invalid index JSON") from exc

            self._validate_index()
        except Exception:
            if self._file is not None:
                self._file.close()
                self._file = None
            raise

        compression = self.metadata["compression"]
        self._decompressor = compression == "zlib"

    def _validate_metadata(self, metadata_bytes: bytes):
        if not isinstance(self.metadata, dict):
            raise InvalidAIDATAFile("Metadata must be a JSON object")

        required = {
            "format", "version", "samples", "x_shape", "y_shape",
            "x_dtype", "y_dtype", "x_ndim", "y_ndim", "compression",
            "chunk_size", "checksum", "metadata_sha256",
        }
        missing = required.difference(self.metadata)
        if missing:
            raise InvalidAIDATAFile(f"Metadata missing required fields: {sorted(missing)}")

        if self.metadata["format"] != "AIDATA":
            raise InvalidAIDATAFile("Invalid metadata format")
        if int(self.metadata["version"]) != VERSION:
            raise InvalidAIDATAFile("Metadata version does not match header")
        if self.metadata["checksum"] != "crc32":
            raise InvalidAIDATAFile("Unsupported checksum algorithm")
        if self.metadata["compression"] not in ("zlib", "none"):
            raise InvalidAIDATAFile("Unsupported compression")
        try:
            compression_level = int(self.metadata.get("compression_level", 0))
        except Exception as exc:
            raise InvalidAIDATAFile("Invalid compression_level") from exc
        if self.metadata["compression"] == "zlib" and not 1 <= compression_level <= 9:
            raise InvalidAIDATAFile("Invalid zlib compression level")
        if self.metadata["compression"] == "none" and compression_level != 0:
            raise InvalidAIDATAFile("Compression level must be 0 when compression is disabled")

        try:
            samples = int(self.metadata["samples"])
            chunk_size = int(self.metadata["chunk_size"])
            x_shape = tuple(int(v) for v in self.metadata["x_shape"])
            y_shape = tuple(int(v) for v in self.metadata["y_shape"])
            x_dtype = np.dtype(self.metadata["x_dtype"])
            y_dtype = np.dtype(self.metadata["y_dtype"])
        except Exception as exc:
            raise InvalidAIDATAFile("Invalid metadata types") from exc

        if samples < 0 or chunk_size <= 0:
            raise InvalidAIDATAFile("Invalid samples or chunk_size")
        if len(x_shape) < 2 or len(y_shape) < 1:
            raise InvalidAIDATAFile("Invalid array dimensions")
        if x_shape[0] != samples or y_shape[0] != samples:
            raise InvalidAIDATAFile("Metadata shapes do not match sample count")
        if any(v < 0 for v in x_shape + y_shape):
            raise InvalidAIDATAFile("Negative dimensions are not allowed")
        if samples > 0 and (any(v == 0 for v in x_shape[1:]) or any(v == 0 for v in y_shape[1:])):
            raise InvalidAIDATAFile("Non-empty datasets cannot have zero-sized sample dimensions")
        if x_dtype.hasobject or y_dtype.hasobject or x_dtype.fields or y_dtype.fields:
            raise InvalidAIDATAFile("Unsupported dtype in metadata")
        if not x_dtype.isnative or not y_dtype.isnative:
            raise InvalidAIDATAFile("AIDATA requires native-endian dtypes")
        if int(self.metadata["x_ndim"]) != len(x_shape) or int(self.metadata["y_ndim"]) != len(y_shape):
            raise InvalidAIDATAFile("ndim does not match shape")

        # Verify metadata hash after removing the hash field itself.
        supplied_hash = str(self.metadata["metadata_sha256"])
        canonical = dict(self.metadata)
        canonical.pop("metadata_sha256", None)
        expected_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        if supplied_hash != expected_hash:
            raise InvalidAIDATAFile("Metadata SHA-256 mismatch")

    def _validate_index(self):
        if not isinstance(self.index, list) or len(self.index) != self.chunk_count:
            raise InvalidAIDATAFile("Chunk count does not match index")

        samples = int(self.metadata["samples"])
        chunk_size = int(self.metadata["chunk_size"])
        x_dtype = np.dtype(self.metadata["x_dtype"])
        y_dtype = np.dtype(self.metadata["y_dtype"])
        x_per_sample = int(np.prod(self.metadata["x_shape"][1:], dtype=np.int64)) * x_dtype.itemsize
        y_per_sample = int(np.prod(self.metadata["y_shape"][1:], dtype=np.int64)) * y_dtype.itemsize

        expected_start = 0
        previous_end = self._data_start
        for i, c in enumerate(self.index):
            if not isinstance(c, dict):
                raise InvalidAIDATAFile(f"Chunk {i} is not an object")
            required = (
                "start", "end", "x_offset", "y_offset", "x_size", "y_size",
                "x_raw_size", "y_raw_size", "x_crc32", "y_crc32",
            )
            if any(k not in c for k in required):
                raise InvalidAIDATAFile(f"Chunk {i} is missing required fields")
            try:
                start, end = int(c["start"]), int(c["end"])
                xo, yo = int(c["x_offset"]), int(c["y_offset"])
                xs, ys = int(c["x_size"]), int(c["y_size"])
                xrs, yrs = int(c["x_raw_size"]), int(c["y_raw_size"])
            except Exception as exc:
                raise InvalidAIDATAFile(f"Chunk {i} contains invalid numeric fields") from exc

            if start != expected_start or end <= start:
                raise InvalidAIDATAFile(f"Invalid sample range in chunk {i}")
            if end - start > chunk_size:
                raise InvalidAIDATAFile(f"Chunk {i} exceeds configured chunk_size")
            if min(xo, yo, xs, ys, xrs, yrs) < 0:
                raise InvalidAIDATAFile(f"Negative byte value in chunk {i}")
            try:
                x_crc = int(c["x_crc32"])
                y_crc = int(c["y_crc32"])
            except Exception as exc:
                raise InvalidAIDATAFile(f"Invalid checksum in chunk {i}") from exc
            if not (0 <= x_crc <= 0xFFFFFFFF and 0 <= y_crc <= 0xFFFFFFFF):
                raise InvalidAIDATAFile(f"Checksum out of range in chunk {i}")
            if xo != previous_end or yo != xo + xs:
                raise InvalidAIDATAFile(f"Chunk {i} has invalid/non-contiguous byte ranges")
            if xo + xs != yo or yo + ys > self._index_offset:
                raise InvalidAIDATAFile(f"Chunk {i} overlaps the index")

            expected_x_raw = (end - start) * x_per_sample
            expected_y_raw = (end - start) * y_per_sample
            if xrs != expected_x_raw or yrs != expected_y_raw:
                raise InvalidAIDATAFile(f"Raw byte size mismatch in chunk {i}")
            if xs == 0 or ys == 0:
                raise InvalidAIDATAFile(f"Empty payload in chunk {i}")

            previous_end = yo + ys
            expected_start = end

        if expected_start != samples:
            raise InvalidAIDATAFile("Index sample count does not match metadata")
        if samples == 0 and self.chunk_count != 0:
            raise InvalidAIDATAFile("Empty dataset must not contain chunks")
        if samples > 0 and self.chunk_count == 0:
            raise InvalidAIDATAFile("Non-empty dataset has no chunks")

    def __len__(self):
        return int(self.metadata["samples"])

    def info(self):
        return dict(self.metadata)

    @staticmethod
    def _decompress_bounded(data: bytes, expected_size: int) -> bytes:
        obj = zlib.decompressobj()
        out = obj.decompress(data, expected_size + 1)
        if len(out) > expected_size or obj.unconsumed_tail:
            raise InvalidAIDATAFile("Decompressed payload exceeds expected size")
        out += obj.flush()
        if len(out) != expected_size:
            raise InvalidAIDATAFile("Unexpected decompressed payload size")
        if not obj.eof:
            raise InvalidAIDATAFile("Incomplete compressed payload")
        if obj.unused_data:
            raise InvalidAIDATAFile("Trailing data after compressed payload")
        return out

    def _read_chunk(self, chunk_id):
        if self._file is None:
            raise ValueError("Reader is closed")
        if chunk_id in self._cache:
            self.cache_hits += 1
            value = self._cache.pop(chunk_id)
            self._cache[chunk_id] = value
            return value

        self.cache_misses += 1
        c = self.index[chunk_id]
        self._file.seek(c["x_offset"])
        x_data = self._file.read(c["x_size"])
        y_data = self._file.read(c["y_size"])
        if len(x_data) != c["x_size"] or len(y_data) != c["y_size"]:
            raise InvalidAIDATAFile(f"Truncated data in chunk {chunk_id}")

        try:
            if self._decompressor:
                x_data = self._decompress_bounded(x_data, c["x_raw_size"])
                y_data = self._decompress_bounded(y_data, c["y_raw_size"])
        except Exception as exc:
            raise InvalidAIDATAFile(f"Failed to decompress chunk {chunk_id}: {exc}") from exc

        if len(x_data) != c["x_raw_size"] or len(y_data) != c["y_raw_size"]:
            raise InvalidAIDATAFile(f"Unexpected decompressed size in chunk {chunk_id}")
        if zlib.crc32(x_data) & 0xFFFFFFFF != int(c["x_crc32"]):
            raise InvalidAIDATAFile(f"X checksum mismatch in chunk {chunk_id}")
        if zlib.crc32(y_data) & 0xFFFFFFFF != int(c["y_crc32"]):
            raise InvalidAIDATAFile(f"Y checksum mismatch in chunk {chunk_id}")

        try:
            n = int(c["end"] - c["start"])
            x_shape = (n, *tuple(int(v) for v in self.metadata["x_shape"][1:]))
            y_shape = (n, *tuple(int(v) for v in self.metadata["y_shape"][1:]))
            X = np.frombuffer(x_data, dtype=np.dtype(self.metadata["x_dtype"])).reshape(x_shape)
            y = np.frombuffer(y_data, dtype=np.dtype(self.metadata["y_dtype"])).reshape(y_shape)
        except Exception as exc:
            raise InvalidAIDATAFile(f"Invalid array payload in chunk {chunk_id}") from exc

        result = (X, y)
        self._cache[chunk_id] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("AIDATA index must be an integer (got {!r})".format(type(index).__name__))
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("AIDATA index out of range: index={}, samples={}".format(index, len(self)))
        chunk_id = min(index // int(self.metadata["chunk_size"]), self.chunk_count - 1)
        X, y = self._read_chunk(chunk_id)
        local = index - int(self.index[chunk_id]["start"])
        return X[local], y[local]

    def get_batch(self, start, batch_size):
        if not isinstance(start, (int, np.integer)) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError("start and batch_size must be integers")
        start, batch_size = int(start), int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0 (got {!r})".format(batch_size))
        if start < 0:
            start += len(self)
        if start < 0 or start >= len(self):
            raise IndexError("batch start is out of range: start={}, samples={}".format(start, len(self)))

        end = min(start + batch_size, len(self))
        first = start // int(self.metadata["chunk_size"])
        last = (end - 1) // int(self.metadata["chunk_size"])
        parts_x, parts_y = [], []
        for cid in range(first, last + 1):
            Xc, yc = self._read_chunk(cid)
            c = self.index[cid]
            a = max(start, int(c["start"])) - int(c["start"])
            b = min(end, int(c["end"])) - int(c["start"])
            parts_x.append(Xc[a:b])
            parts_y.append(yc[a:b])

        if len(parts_x) == 1:
            return parts_x[0], parts_y[0]
        return np.concatenate(parts_x, axis=0), np.concatenate(parts_y, axis=0)

    def get_chunk(self, chunk_id):
        if not isinstance(chunk_id, (int, np.integer)):
            raise TypeError("chunk_id must be an integer")
        chunk_id = int(chunk_id)
        if chunk_id < 0:
            chunk_id += self.chunk_count
        if chunk_id < 0 or chunk_id >= self.chunk_count:
            raise IndexError("Chunk index out of range")
        return self._read_chunk(chunk_id)

    def cache_info(self):
        return {
            "cache_size": self.cache_size,
            "cached_chunks": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_keys": list(self._cache.keys()),
        }

    def clear_cache(self):
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def close(self):
        self.clear_cache()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
