# 6. docs/FAQ.md
# ❓ Frequently Asked Questions

---

## General Questions

### Q: What is AIDATA?

AIDATA is a compressed, indexed dataset format for machine learning. It stores NumPy arrays as chunks on disk, lets you read arbitrary samples or batches instantly, and integrates directly with PyTorch.

### Q: When should I use AIDATA instead of NumPy/CSV?

Use AIDATA when:
- Your dataset is too large to fit in RAM
- You need fast random/batch access without loading everything
- You want built-in compression to save disk space
- You want metadata stored with the data
- You train with PyTorch and want zero boilerplate loading

Use NumPy when:
- Your dataset is small (< 100 MB) and fits in RAM
- You need maximum read speed and don\'t care about compression

Use CSV when:
- Humans need to read the file
- You need Excel compatibility

### Q: Is AIDATA faster than NumPy?

- **First read:** NumPy is faster (no decompression)
- **Repeated reads:** AIDATA is faster (LRU cache + index)
- **Random access:** AIDATA is much faster (no full load)
- **Disk usage:** AIDATA is 2×–5× smaller

### Q: Can I use AIDATA without PyTorch?

Yes! `AIDATAWriter` and `AIDATAReader` are pure NumPy. PyTorch integration is optional.

```python
from aidata import AIDATAReader

reader = AIDATAReader("data.aidata")
X_batch, y_batch = reader.get_batch(0, 1000)  # Pure NumPy arrays
```

---

## Installation & Setup

### Q: Installation fails with "No module named zstandard"

```bash
pip install zstandard
```

Or install with all dependencies:
```bash
pip install aidata
```

### Q: "ImportError: cannot import name AIDATAWriter"

You haven\'t installed the package. Run:
```bash
cd aidata
pip install -e .
```

### Q: Does AIDATA work on Windows/Mac/Linux?

Yes. All dependencies (NumPy, PyTorch, zstandard) are cross-platform.

---

## Usage Questions

### Q: How do I choose chunk_size?

| Dataset Size | Recommended chunk_size |
|-------------|----------------------|
| < 10K samples | 256–1024 |
| 10K–100K | 1024–4096 |
| 100K–1M | 4096–16384 |
| > 1M | 16384–65536 |

**Rules:**
- Larger chunks = better compression, faster sequential reads
- Smaller chunks = better random access, less RAM per chunk
- Default `4096` works well for most cases

### Q: What should I set cache_size to?

Set `cache_size` to the number of chunks you expect to reuse:

```python
# If your batch_size is 256 and chunk_size is 4096,
# each batch touches ~1 chunk.
# For 3 epochs of reuse, cache_size=3 is enough.

# For shuffled training, set cache_size higher:
reader = AIDATAReader("data.aidata", cache_size=16)
```

### Q: Can I append data to an existing file?

Not yet. Currently you must rewrite the entire file. Incremental append is on the roadmap.

**Workaround:**
```python
# Read old data, concatenate, rewrite
reader = AIDATAReader("old.aidata")
X_old, y_old = reader.get_batch(0, len(reader))

X_new = np.concatenate([X_old, X_additional])
y_new = np.concatenate([y_old, y_additional])

writer = AIDATAWriter("new.aidata")
writer.write(X_new, y_new)
```

### Q: Can I store images or multi-dimensional targets?

Currently AIDATA supports:
- `X`: 2D array `(samples, features)`
- `y`: 1D array `(samples,)`

Multi-dimensional targets (e.g., segmentation masks `(samples, H, W)`) are planned for v0.6.0.

**Workaround:** Flatten your target, store shape in metadata, reshape on read:
```python
# Write
y_flat = y.reshape(len(y), -1)
writer.write(X, y_flat, metadata={"y_shape": list(y.shape)})

# Read
reader = AIDATAReader("data.aidata")
X_batch, y_batch = reader.get_batch(0, 100)
y_batch = y_batch.reshape(100, *reader.metadata["y_shape"][1:])
```

---

## Performance Questions

### Q: Training is slow. How do I speed it up?

1. **Increase chunk_size** — reduces disk seeks
2. **Increase cache_size** — keeps more chunks in RAM
3. **Use AIDATALoader instead of DataLoader** — direct chunk-aligned reads
4. **Set device in AIDATALoader** — avoids manual `.to(device)`
5. **Profile first** — run `profile_training.py` to find the bottleneck

### Q: My GPU is idle while CPU loads data

Use prefetching (coming in v0.6.0) or increase `num_workers` with `AIDATABatchDataset`:

```python
from torch.utils.data import DataLoader
from aidata import AIDATABatchDataset

ds = AIDATABatchDataset("data.aidata", batch_size=256)
loader = DataLoader(ds, batch_size=None, num_workers=4)
```

### Q: File size is still too big

- Try larger `chunk_size` (better compression ratio)
- Ensure your data has patterns (random data doesn\'t compress well)
- Consider quantizing to `float16` before writing:
  ```python
  X = X.astype(np.float16)
  ```

---

## Error Messages

### Q: "Invalid AIDATA magic"

The file is not a valid AIDATA file. Common causes:
- File was corrupted during transfer
- You\'re opening a NumPy/CSV file with `.aidata` extension
- Incomplete write (program crashed)

**Fix:** Re-create the file with `AIDATAWriter`.

### Q: "Unsupported version"

The file was created with a newer/older version of AIDATA. Check:
```python
import aidata
print(aidata.__version__)
```

**Fix:** Recreate the file with the current library version.

### Q: "Batch start out of range"

You requested a batch starting beyond the dataset:
```python
reader.get_batch(start=100000, batch_size=256)  # If dataset has < 100000 samples
```

**Fix:** Check `len(reader)` first.

### Q: "X must be a 2D array"

Your features array has the wrong shape:
```python
X = np.random.rand(1000)       # ❌ 1D
X = np.random.rand(1000, 10)   # ✅ 2D
```

---

## Comparison Questions

### Q: AIDATA vs HDF5?

| | AIDATA | HDF5 |
|---|--------|------|
| Dependencies | Light (zstd) | Heavy (h5py, C libs) |
| PyTorch | Native | Requires wrapper |
| Cache | Built-in LRU | Manual |
| Compression | zstd per chunk | Optional, full-file |
| Index | JSON footer | B-tree |

Use HDF5 for complex nested data. Use AIDATA for simple, fast ML datasets.

### Q: AIDATA vs TFRecord?

TFRecord is TensorFlow-specific. AIDATA is PyTorch-first but framework-agnostic.

### Q: AIDATA vs Parquet?

Parquet is columnar and great for analytics. AIDATA is row-oriented and optimized for ML training loops.

---

## Contributing & Support

### Q: How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md). We welcome:
- Bug reports
- Feature requests
- Documentation improvements
- Code contributions

### Q: Where do I report bugs?

Open an issue on GitHub with:
- AIDATA version
- Python version
- Minimal code to reproduce
- Full error traceback

### Q: Is there a Discord/Slack community?

Not yet. Use GitHub Discussions for now.
'''