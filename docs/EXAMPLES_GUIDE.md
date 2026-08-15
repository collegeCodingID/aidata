# 4. docs/EXAMPLES_GUIDE.md
examples_guide = '''# 📖 Examples Guide

Detailed walkthrough of every example in the `examples/` folder.

---

## 1. `basic.py` — Read/Write Basics

**What it teaches:** Writing a file, reading samples, batches, chunks, and cache behavior.

```bash
cd examples
python basic.py
```

### Key Concepts

**Writing with metadata:**
```python
writer = AIDATAWriter("training.aidata")
writer.write(
    X, y,
    metadata={
        "dataset_name": "Binary Classification",
        "task": "binary_classification",
    },
    compression=True,
    chunk_size=4096,
)
```

**Random access:**
```python
reader = AIDATAReader("training.aidata", cache_size=4)
sample = reader[5000]        # Single sample
batch = reader.get_batch(5000, 256)  # Batch
```

**Cache demonstration:**
```python
reader.get_chunk(0)          # Cache MISS
reader.get_chunk(0)          # Cache HIT — instant!
```

**PyTorch integration:**
```python
from aidata.integrations import AIDATAPyTorchDataset, AIDATABatchDataset

torch_ds = AIDATAPyTorchDataset(reader)      # Sample-level
batch_ds = AIDATABatchDataset(reader, 256)   # Batch-level
```

---

## 2. `train_test.py` — Standard PyTorch DataLoader

**What it teaches:** Using `AIDATABatchDataset` with `torch.utils.data.DataLoader`.

```bash
python train_test.py
```

### Key Pattern

```python
from torch.utils.data import DataLoader
from aidata import AIDATAReader
from aidata.integrations import AIDATABatchDataset

reader = AIDATAReader("training.aidata")
batch_ds = AIDATABatchDataset(reader, batch_size=256)

# batch_size=None because dataset already returns batches!
loader = DataLoader(batch_ds, batch_size=None, shuffle=True)

for X_batch, y_batch in loader:
    # Training loop
    pass
```

**When to use this:**
- You need `DataLoader` features (multi-worker, custom sampler, pin_memory)
- You already have PyTorch training code and want minimal changes

---

## 3. `pytorch_train.py` — Native AIDATA Loader

**What it teaches:** Using `AIDATALoader` for maximum performance.

```bash
python pytorch_train.py
```

### Key Pattern

```python
from aidata import AIDATALoader

loader = AIDATALoader(
    path="training.aidata",
    batch_size=256,
    shuffle=True,
    cache_size=8,
    device="cuda",      # Auto GPU transfer!
    seed=42,
)

for X_batch, y_batch in loader:
    # X_batch is already on CUDA!
    logits = model(X_batch)
```

**Advantages over DataLoader:**
- No need to write custom `collate_fn`
- Direct chunk-aligned reading (faster)
- Built-in device transfer
- Seed-controlled shuffle

---

## 4. `pytorch_dataloader.py` — IterableDataset Style

**What it teaches:** Using AIDATA as a PyTorch `IterableDataset`.

```bash
python pytorch_dataloader.py
```

### Key Pattern

```python
from torch.utils.data import IterableDataset, DataLoader
from aidata import AIDATAReader

class AIDATADataset(IterableDataset):
    def __iter__(self):
        reader = AIDATAReader(self.file_path)
        chunk_index = 0
        while True:
            try:
                chunk = reader.get_chunk(chunk_index)
            except IndexError:
                break
            yield self._normalize(chunk)
            chunk_index += 1
```

**When to use this:**
- Streaming large datasets
- Custom chunk processing logic
- When you want full control over iteration

---

## 5. `benchmark.py` — Read Performance

**What it teaches:** How chunk size affects read speed.

```bash
python benchmark.py
```

### What it measures

| Metric | Description |
|--------|-------------|
| Write time | How long to create the file |
| Open time | How long to read header + index |
| Random | Time to read 1 random sample |
| Batch | Time to read 1 random batch |
| Sequential | Time to read all batches |
| Full | Time to read all chunks |

### Typical Results

```
Chunk | Chunks | Size  | Write  | Open    | Random  | Batch   | Sequential | Full
------|--------|-------|--------|---------|---------|---------|------------|------
  256 |    391 | 15.2  | 0.5234 | 0.000123| 0.000234| 0.000456| 0.1234     | 0.2345
 1024 |     98 | 14.8  | 0.4123 | 0.000112| 0.000198| 0.000389| 0.0987     | 0.1987
 4096 |     25 | 14.5  | 0.3456 | 0.000105| 0.000187| 0.000345| 0.0876     | 0.1765
16384 |      7 | 14.3  | 0.2987 | 0.000098| 0.000176| 0.000298| 0.0765     | 0.1543
```

**Rule of thumb:** Larger chunks = faster sequential reads, but more RAM usage.

---

## 6. `training_benchmark.py` — End-to-End Comparison

**What it teaches:** How AIDATA compares to CSV and NumPy for training.

```bash
python training_benchmark.py
```

### Comparison

| Format | File Size | Write Time | Load Time | Training Time | Samples/sec |
|--------|-----------|------------|-----------|---------------|-------------|
| CSV    | ~45 MB    | Slow       | Slow      | Medium        | ~50,000     |
| NumPy  | ~38 MB    | Fast       | Fast      | Fast          | ~80,000     |
| AIDATA | ~15 MB    | Medium     | Instant   | Fast          | ~75,000     |

**Key takeaway:** AIDATA gives you **compression + instant loading** with **minimal training overhead**.

---

## 7. `profile_training.py` — Read vs Compute Time

**What it teaches:** Whether data loading or GPU computation is your bottleneck.

```bash
python profile_training.py
```

### Output Example

```
Epoch 1: read=0.0234s | compute=1.2345s
Epoch 2: read=0.0198s | compute=1.1987s
Epoch 3: read=0.0212s | compute=1.2456s

PROFILE RESULT
Total time:       3.7654s
AIDATA read time:  0.0644s
PyTorch compute:   3.6788s
Other overhead:    0.0222s
Samples/sec: 234,567
```

**Interpretation:**
- If `read` >> `compute` → Increase cache_size, use prefetching, or larger chunks
- If `compute` >> `read` → Your data loading is optimal!

---

## 🎯 Choosing the Right Example

| Your Goal | Start With |
|-----------|-----------|
| Just see how it works | `basic.py` |
| Integrate with existing PyTorch code | `train_test.py` |
| Maximum training performance | `pytorch_train.py` |
| Stream/process chunks | `pytorch_dataloader.py` |
| Tune chunk size | `benchmark.py` |
| Compare with CSV/NumPy | `training_benchmark.py` |
| Find bottlenecks | `profile_training.py` |
'''

