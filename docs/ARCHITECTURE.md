# 5. docs/ARCHITECTURE.md
# 🏗️ Architecture & File Format

Understanding how AIDATA stores data on disk.

---

## File Layout

An `.aidata` file is a single binary file with 5 sections:

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER (16 bytes)                                          │
│  - Magic: "AIDT" (4 bytes)                                  │
│  - Version: 1 (4 bytes)                                     │
│  - Metadata size (4 bytes)                                  │
│  - Chunk count (4 bytes)                                    │
├─────────────────────────────────────────────────────────────┤
│  METADATA (JSON)                                            │
│  - samples, features, dtypes, compression, chunk_size, etc. │
│  - Plus any user-defined metadata                           │
├─────────────────────────────────────────────────────────────┤
│  CHUNKS (binary, compressed)                                │
│  Chunk 0 X data → Chunk 0 Y data → Chunk 1 X data → ...     │
│  Each chunk is independently compressed with zstd           │
├─────────────────────────────────────────────────────────────┤
│  INDEX (JSON)                                               │
│  Array of chunk descriptors:                                │
│  [{start, end, x_offset, y_offset, x_size, y_size, ...}, ] │
├─────────────────────────────────────────────────────────────┤
│  FOOTER (12 bytes)                                          │
│  - Index offset (8 bytes)                                   │
│  - Index size (4 bytes)                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Layout?

### 1. Header First
- Instant validation: check magic + version in first 16 bytes
- Know metadata size immediately → jump to metadata

### 2. Metadata Early
- Read dataset properties (features, dtype, compression) without scanning
- User metadata accessible instantly

### 3. Chunks in Middle
- Main payload. X and Y stored separately so you can read only what you need
- Each chunk compressed independently → random access possible

### 4. Index Before Footer
- Footer is fixed 12 bytes at the very end
- Seek to `-12` from end → read index_offset → seek to index
- No need to scan the entire file!

### 5. Footer Last
- Fixed size makes it trivial to locate
- Contains pointer to index

---

## Chunk Structure

Each chunk contains a contiguous slice of samples:

```python
# Chunk i covers samples [start, end)
chunk = {
    "start": 4096,        # First sample index
    "end": 8192,          # Last sample index (exclusive)
    "x_offset": 12345,    # Byte position of X data in file
    "y_offset": 23456,    # Byte position of Y data in file
    "x_size": 1024,       # Compressed X size in bytes
    "y_size": 512,        # Compressed Y size in bytes
    "x_raw_size": 65536,  # Decompressed X size (for zstd)
    "y_raw_size": 16384,  # Decompressed Y size (for zstd)
}
```

### Why Separate X and Y?

- Some workflows only need features (inference)
- Some only need labels (analysis)
- Separate storage allows future partial reads

---

## Compression Strategy

**Algorithm:** `zstd` (Zstandard) level 3

**Why zstd?**
- Faster than gzip/bzip2
- Better ratio than lz4
- Widely supported, stable

**Per-chunk compression:**
- Each chunk compressed independently
- Trade-off: slightly worse ratio than full-file compression
- Benefit: random access without decompressing everything

**Typical compression ratios:**
- Random float32 data: ~1.5×
- Structured/tabular data: ~3×–5×
- Sparse/integer data: ~5×–10×

---

## LRU Cache Strategy

```python
from collections import OrderedDict

class AIDATAReader:
    def __init__(self, cache_size=8):
        self._cache = OrderedDict()  # {chunk_id: (X, y)}
```

### How it works

1. **Access chunk:**
   - If in cache → **HIT** → move to end (most recently used)
   - If not in cache → **MISS** → decompress from disk, add to cache

2. **Eviction:**
   - When cache exceeds `cache_size`, remove oldest item (`popitem(last=False)`)

3. **Cache info:**
   ```python
   reader.cache_info()
   # {
   #   "cache_size": 8,
   #   "cached_chunks": 3,
   #   "cache_hits": 45,
   #   "cache_misses": 12
   # }
   ```

### Tuning cache_size

| Dataset Size | chunk_size | Recommended cache_size |
|-------------|------------|----------------------|
| < 100K samples | 4096 | 4–8 |
| 100K–1M samples | 4096 | 8–16 |
| 1M–10M samples | 16384 | 16–32 |
| > 10M samples | 65536 | 32–64 |

Rule: `cache_size` should cover at least 2–3 epochs of chunk reuse.

---

## Memory Footprint

For a dataset with:
- `N` samples
- `F` features
- `C` chunk_size
- `cache_size = K`

**Peak RAM usage:**
```
~ K × C × F × 4 bytes (float32 X)
+ K × C × 8 bytes (int64 y)
+ file overhead (index + metadata, usually < 1 MB)
```

**Example:**
- 1M samples, 128 features, chunk_size=4096, cache_size=8
- RAM = 8 × 4096 × 128 × 4 + 8 × 4096 × 8 ≈ **16.8 MB**
- Compare: Full NumPy load = 1M × 128 × 4 = **512 MB**

**AIDATA uses ~30× less RAM!**

---

## Comparison with Other Formats

| Feature | AIDATA | NumPy (.npy) | HDF5 | Parquet | TFRecord |
|---------|--------|-------------|------|---------|----------|
| Compression | ✅ zstd | ❌ No | ✅ Optional | ✅ zstd | ❌ No |
| Random access | ✅ Yes | ❌ Full load | ✅ Yes | ⚠️ Slow | ⚠️ Sequential |
| Chunked | ✅ Yes | ❌ No | ✅ Yes | ⚠️ Row groups | ❌ No |
| PyTorch native | ✅ Yes | ⚠️ Manual | ⚠️ h5py | ⚠️ pandas | ⚠️ TF only |
| Metadata | ✅ JSON | ❌ No | ✅ Yes | ✅ Yes | ❌ Limited |
| Index | ✅ Footer | ❌ No | ✅ B-tree | ❌ No | ❌ No |
| LRU cache | ✅ Built-in | ❌ No | ❌ No | ❌ No | ❌ No |

---

## Future Format Improvements

- **Checksums per chunk** (CRC32) for corruption detection
- **Variable-length dtypes** (strings, images)
- **Delta encoding** for sequential data
- **Encryption** support for sensitive data
'''