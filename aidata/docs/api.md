# API reference

This page documents the intended public API in AIDATA 0.5.6.

## `AIDATAWriter`

```python
from aidata import AIDATAWriter
```

Constructor:

```python
AIDATAWriter(path)
```

Main method:

```python
writer.write(
    X,
    y,
    metadata=None,
    compression=True,
    chunk_size=4096,
    compression_level=3,
    verbose=True,
)
```

Parameters:

- `X`: NumPy-compatible array with at least 2 dimensions.
- `y`: NumPy-compatible array with at least 1 dimension.
- `metadata`: optional JSON-serializable dictionary.
- `compression`: boolean; currently `zlib` or no compression.
- `chunk_size`: positive number of samples per chunk.
- `compression_level`: zlib level `1..9` when compression is enabled.
- `verbose`: print write information when true.

## `AIDATAReader`

```python
from aidata import AIDATAReader
```

Constructor:

```python
AIDATAReader(path, cache_size=8)
```

Important operations:

```python
len(reader)
reader[index]
reader.get_batch(start, batch_size)
reader.metadata
reader.index
reader.cache_hits
reader.cache_misses
reader.close()
```

Use it as a context manager.

## `AIDATADataset`

```python
from aidata import AIDATADataset
```

A PyTorch map-style dataset backed by an AIDATA file. It is designed so readers are created locally inside worker processes instead of sharing an open file handle across workers.

Requires the optional PyTorch dependency.

## `AIDATABatchDataset`

```python
from aidata import AIDATABatchDataset
```

Exposes contiguous batches as dataset items. This can be useful when the training loop should operate on pre-batched samples.

## `AIDATALoader`

```python
from aidata import AIDATALoader
```

A context-managed batch loader for sequential/chunk-oriented training access.

When `shuffle=True`, batch order is shuffled. Individual samples are not independently shuffled by this loader.

## Exceptions

```python
from aidata.exceptions import (
    AIDATAError,
    InvalidAIDATAFile,
    UnsupportedVersion,
)
```

Inheritance:

```text
Exception
└── AIDATAError
    ├── InvalidAIDATAFile
    └── UnsupportedVersion
```

Use `InvalidAIDATAFile` when diagnosing a dataset file and `AIDATAError` for general AIDATA-specific failures.
