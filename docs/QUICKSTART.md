# 2. docs/QUICKSTART.md
quickstart = '''# 🚀 Quick Start Guide

This guide will take you from zero to training in under 5 minutes.

---

## Step 1: Install AIDATA

```bash
pip install aidata
```

If you cloned the repo:

```bash
cd aidata
pip install -e .
```

Verify installation:

```bash
python -c "import aidata; print(aidata.__version__)"
# Output: 0.5.0
```

---

## Step 2: Create Your First Dataset

Create a Python file called `first_dataset.py`:

```python
import numpy as np
from aidata import AIDATAWriter

# Create sample data
X = np.random.rand(50_000, 20).astype(np.float32)
y = np.random.randint(0, 2, size=50_000, dtype=np.int64)

# Write to AIDATA format
writer = AIDATAWriter("my_data.aidata")
writer.write(
    X,
    y,
    metadata={
        "dataset_name": "My First Dataset",
        "task": "binary_classification",
        "author": "Your Name",
    },
    compression=True,
    chunk_size=4096,
)
```

Run it:

```bash
python first_dataset.py
```

You will see:
```
File created : my_data.aidata
Samples      : 50000
Features     : 20
Chunks       : 13
Chunk size   : 4096
Compression  : zstd
```

---

## Step 3: Read the Dataset

Create `read_data.py`:

```python
from aidata import AIDATAReader

# Open the file
reader = AIDATAReader("my_data.aidata", cache_size=4)

# Check info
print("Total samples:", len(reader))
print("Features:", reader.metadata["features"])
print("Compression:", reader.metadata["compression"])
print("Custom metadata:", reader.metadata.get("dataset_name"))

# Read a single sample
sample_x, sample_y = reader[100]
print("Sample shape:", sample_x.shape, "Label:", sample_y)

# Read a batch
X_batch, y_batch = reader.get_batch(start=1000, batch_size=256)
print("Batch shape:", X_batch.shape, y_batch.shape)

# Check cache performance
print(reader.cache_info())

reader.close()
```

---

## Step 4: Train with PyTorch

Create `train.py`:

```python
import torch
from torch import nn
from aidata import AIDATALoader

# Native AIDATA loader — already returns torch.Tensor!
loader = AIDATALoader(
    "my_data.aidata",
    batch_size=256,
    shuffle=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Simple model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(loader.device if hasattr(loader, "device") else "cpu")

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):
    total_loss = 0.0
    for X_batch, y_batch in loader:
        logits = model(X_batch).squeeze(1)
        loss = loss_fn(logits, y_batch.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

loader.close()
```

---

## Step 5: Use with Standard PyTorch DataLoader

If you prefer `torch.utils.data.DataLoader`:

```python
from torch.utils.data import DataLoader
from aidata import AIDATABatchDataset

# Each item is already a full batch
dataset = AIDATABatchDataset("my_data.aidata", batch_size=256)

# batch_size=None because dataset already returns batches
loader = DataLoader(dataset, batch_size=None, shuffle=True)

for X_batch, y_batch in loader:
    # Training code here
    pass
```

---

## ✅ What You Learned

| Concept | What It Does |
|---------|-------------|
| `AIDATAWriter` | Saves NumPy arrays as compressed, indexed files |
| `AIDATAReader` | Reads single samples, batches, or full chunks |
| `AIDATALoader` | Native PyTorch iterator with shuffle, cache, GPU transfer |
| `AIDATABatchDataset` | PyTorch `Dataset` compatible with `DataLoader` |
| `cache_size` | Keeps recent chunks in RAM for faster re-access |

---

## 🎯 Next Steps

- Read the [API Reference](API_REFERENCE.md) for all options
- Check [Examples Guide](EXAMPLES_GUIDE.md) for advanced usage
- See [Architecture](ARCHITECTURE.md) to understand the file format
'''

with open(f"{base}/docs/QUICKSTART.md", "w") as f:
    f.write(quickstart)

