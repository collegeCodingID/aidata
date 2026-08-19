# Getting started

## 1. Install

```bash
pip install aidata
```

For PyTorch:

```bash
pip install "aidata[torch]"
```

## 2. Write a dataset

```python
import numpy as np
from aidata import AIDATAWriter

X = np.random.randn(10_000, 32).astype(np.float32)
y = np.random.randint(0, 5, size=10_000, dtype=np.int64)

AIDATAWriter("train.aidata").write(
    X,
    y,
    compression=True,
    compression_level=3,
    chunk_size=1024,
    metadata={"dataset": "demo", "task": "classification"},
)
```

Metadata must be JSON-serializable and must not use AIDATA's reserved metadata keys.

## 3. Read samples

```python
from aidata import AIDATAReader

with AIDATAReader("train.aidata") as reader:
    print(len(reader))
    print(reader.metadata)

    x, y = reader[10]
    X_batch, y_batch = reader.get_batch(100, 64)
```

## 4. Use the reader as a context manager

Prefer:

```python
with AIDATAReader("train.aidata") as reader:
    ...
```

The reader owns a file handle and should be closed when finished.

## 5. Choose chunk size deliberately

Small chunks:

- better random-access granularity
- more index entries
- potentially more I/O/decompression overhead

Large chunks:

- better sequential throughput in many workloads
- fewer index entries
- more data decompressed per access

Start with `4096`, then benchmark your actual workload.

## 6. Compression

Compression can reduce storage size but adds CPU work.

```python
AIDATAWriter("data.aidata").write(
    X,
    y,
    compression=True,
    compression_level=3,
)
```

Disable it when raw I/O and CPU overhead make that the better trade-off:

```python
AIDATAWriter("data.aidata").write(
    X,
    y,
    compression=False,
)
```

## 7. Common errors

### `InvalidAIDATAFile`

The file is malformed, truncated, inconsistent, or corrupted.

### `UnsupportedVersion`

The file format version is newer or otherwise unsupported by the installed reader.

### `AIDATAError`

AIDATA-specific API or file errors that do not require a more specific exception.

## 8. Source checkout

```bash
git clone https://github.com/YOUR_USERNAME/aidata.git
cd aidata
python -m pip install -e ".[dev]"
pytest -q
PYTHONPATH=src python examples/basic.py
```
