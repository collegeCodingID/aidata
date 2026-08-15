# 3. docs/API_REFERENCE.md
# 📚 API Reference

Complete reference for all public classes and methods in AIDATA.

---

## Table of Contents

- [AIDATAWriter](#aidatawriter)
- [AIDATAReader](#aidatareader)
- [AIDATADataset](#aidatadataset)
- [AIDATABatchDataset](#aidatabatchdataset)
- [AIDATALoader](#aidataloader)
- [Exceptions](#exceptions)

---

## AIDATAWriter

```python
class AIDATAWriter(path)
```

Writes NumPy arrays to a compressed, indexed `.aidata` file.

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Output file path |

### Methods

#### `write(X, y, metadata=None, compression=True, chunk_size=4096, verbose=True)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `array-like` | **required** | Feature matrix. Must be 2D. |
| `y` | `array-like` | **required** | Target vector. Must be 1D. |
| `metadata` | `dict` | `None` | User-defined JSON metadata |
| `compression` | `bool` | `True` | Use `zstd` compression |
| `chunk_size` | `int` | `4096` | Samples per chunk |
| `verbose` | `bool` | `True` | Print summary after writing |

**Raises:**
- `AIDATAError` — If `X` is not 2D, `y` is not 1D, lengths mismatch, or `chunk_size <= 0`

**Example:**
```python
writer = AIDATAWriter("data.aidata")
writer.write(
    X, y,
    metadata={"task": "classification"},
    compression=True,
    chunk_size=4096,
)
```

---

## AIDATAReader

```python
class AIDATAReader(path, cache_size=8)
```

Reads `.aidata` files with random access, batch access, and LRU caching.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | **required** | Path to `.aidata` file |
| `cache_size` | `int` | `8` | Max chunks in LRU cache |

**Raises:**
- `InvalidAIDATAFile` — Corrupted or invalid file
- `UnsupportedVersion` — File version mismatch
- `ValueError` — If `cache_size <= 0`

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `metadata` | `dict` | File header metadata |
| `chunk_count` | `int` | Total number of chunks |
| `version` | `int` | File format version |

### Methods

#### `__len__()` → `int`
Returns total number of samples.

#### `__getitem__(index)` → `(X_sample, y_sample)`
Read a single sample by index. Supports negative indexing.

**Returns:** `(numpy.ndarray, scalar)`

#### `get_batch(start, batch_size)` → `(X_batch, y_batch)`
Read a contiguous batch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `int` | Starting sample index |
| `batch_size` | `int` | Number of samples |

**Returns:** `(numpy.ndarray, numpy.ndarray)`

#### `get_chunk(chunk_id)` → `(X_chunk, y_chunk)`
Read an entire chunk by ID.

#### `info()` → `dict`
Return a copy of metadata.

#### `cache_info()` → `dict`
Return cache statistics:
```python
{
    "cache_size": 8,
    "cached_chunks": 3,
    "cache_hits": 12,
    "cache_misses": 5,
    "cache_keys": [0, 1, 2]
}
```

#### `clear_cache()`
Clear all cached chunks and reset hit/miss counters.

#### `close()`
Release resources.

**Context Manager Support:**
```python
with AIDATAReader("data.aidata") as reader:
    print(len(reader))
```

---

## AIDATADataset

```python
class AIDATADataset(path, cache_size=8, return_tensors=True)
```

PyTorch `Dataset` for **sample-level** access.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | **required** | Path to `.aidata` file |
| `cache_size` | `int` | `8` | Reader cache size |
| `return_tensors` | `bool` | `True` | Return `torch.Tensor` instead of NumPy |

### Methods

#### `__len__()` → `int`
Number of samples.

#### `__getitem__(index)` → `(X, y)`
Get single sample. Returns `torch.Tensor` if `return_tensors=True`.

#### `get_batch(start, batch_size)` → `(X, y)`
Read a contiguous batch.

#### `cache_info()`, `clear_cache()`, `close()`
Delegated to internal reader.

**Context Manager Support:** ✅

---

## AIDATABatchDataset

```python
class AIDATABatchDataset(path, batch_size=256, cache_size=8, return_tensors=True)
```

PyTorch `Dataset` where **each item is a full batch**.

Best used with `DataLoader(batch_size=None)`.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | **required** | Path to `.aidata` file |
| `batch_size` | `int` | `256` | Samples per batch |
| `cache_size` | `int` | `8` | Reader cache size |
| `return_tensors` | `bool` | `True` | Return `torch.Tensor` |

### Methods

#### `__len__()` → `int`
Number of batches (`ceil(n_samples / batch_size)`).

#### `__getitem__(batch_index)` → `(X_batch, y_batch)`
Get a full batch as tensors.

**Example with DataLoader:**
```python
from torch.utils.data import DataLoader
from aidata import AIDATABatchDataset

ds = AIDATABatchDataset("data.aidata", batch_size=256)
loader = DataLoader(ds, batch_size=None, shuffle=True)

for X, y in loader:
    # X.shape == (256, features)
    pass
```

---

## AIDATALoader

```python
class AIDATALoader(path, batch_size=256, shuffle=True, cache_size=8, drop_last=False, device=None, seed=42)
```

Native batch loader. More efficient than `DataLoader` for AIDATA files because it controls chunk reading directly.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | **required** | Path to `.aidata` file |
| `batch_size` | `int` | `256` | Samples per batch |
| `shuffle` | `bool` | `True` | Shuffle batch order each epoch |
| `cache_size` | `int` | `8` | Reader cache size |
| `drop_last` | `bool` | `False` | Skip incomplete last batch |
| `device` | `str` or `torch.device` | `None` | Auto-move tensors to device |
| `seed` | `int` | `42` | Shuffle random seed |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_samples` | `int` | Total samples |
| `num_batches` | `int` | Total batches |
| `epoch` | `int` | Current epoch counter |

### Methods

#### `__len__()` → `int`
Number of batches.

#### `__iter__()` → `Generator`
Yields `(X_batch, y_batch)` tensors. If `device` is set, tensors are already moved.

**Example:**
```python
loader = AIDATALoader("data.aidata", batch_size=256, device="cuda")

for X, y in loader:
    # X is already on CUDA!
    logits = model(X)
```

#### `cache_info()`, `clear_cache()`, `close()`
Cache and resource management.

**Context Manager Support:** ✅

---

## Exceptions

| Exception | When Raised |
|-----------|-------------|
| `AIDATAError` | Base exception for all AIDATA errors |
| `InvalidAIDATAFile` | Corrupted file, bad magic, invalid metadata/index |
| `UnsupportedVersion` | File version does not match library version |

**Example:**
```python
from aidata.exceptions import InvalidAIDATAFile

try:
    reader = AIDATAReader("maybe_corrupt.aidata")
except InvalidAIDATAFile as e:
    print("File is corrupted:", e)
```

---

## Type Hints

AIDATA uses Python 3.8+ type hints:

```python
from typing import Tuple
import numpy as np
import torch

def get_batch(self, start: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

For full type annotations, see the source code in `src/aidata/`.
'''