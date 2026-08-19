# PyTorch integration

Install the optional dependency:

```bash
pip install "aidata[torch]"
```

## Map-style dataset

```python
from torch.utils.data import DataLoader
from aidata import AIDATADataset

dataset = AIDATADataset("train.aidata")
loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
)

for X, y in loader:
    output = model(X)
```

## Worker safety

Do not manually share an already-open `AIDATAReader` between DataLoader workers. `AIDATADataset` is designed around the dataset path and creates worker-local reader state.

## Batch-oriented loading

```python
from aidata import AIDATABatchDataset
from torch.utils.data import DataLoader

batches = AIDATABatchDataset("train.aidata", batch_size=256)
loader = DataLoader(batches, batch_size=None, shuffle=True)
```

## High-throughput sequential access

```python
from aidata import AIDATALoader

with AIDATALoader("train.aidata", batch_size=256, shuffle=True) as loader:
    for X, y in loader:
        loss = train_step(X, y)
```

This loader's shuffle behavior is intentionally batch-oriented. It is not equivalent to a sample-level random sampler.

## Recommended training pattern

For large datasets, benchmark these choices:

1. `AIDATADataset + DataLoader`
2. `AIDATABatchDataset + DataLoader`
3. `AIDATALoader`

Measure end-to-end training throughput, not only file-read speed.

Important variables include `chunk_size`, compression level, batch size, `num_workers`, storage type, and model compute time.
