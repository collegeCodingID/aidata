import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aidata import (
    AIDATAWriter,
    AIDATAReader,
)

from aidata.integrations import (
    AIDATABatchDataset,
)


# ============================================================
# CONFIG
# ============================================================

SAMPLES = 100_000
FEATURES = 20
BATCH_SIZE = 256
EPOCHS = 3


# ============================================================
# SEED
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# CREATE DATA
# ============================================================

print("Creating dataset...")

X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
y = np.random.randint(0, 2, size=SAMPLES, dtype=np.int64)


# ============================================================
# WRITE AIDATA
# ============================================================

print("Creating AIDATA file...")

writer = AIDATAWriter("training.aidata")

writer.write(
    X,
    y,
    metadata={
        "dataset_name": "Training Benchmark",
        "task": "binary_classification",
    },
    compression=True,
    chunk_size=4096,
)


# ============================================================
# READER
# ============================================================

dataset = AIDATAReader("training.aidata", cache_size=8)


# ============================================================
# BATCH DATASET
# ============================================================

batch_dataset = AIDATABatchDataset(dataset, batch_size=BATCH_SIZE)

print()
print("Samples:", len(dataset))
print("Features:", FEATURES)
print("Batches:", len(batch_dataset))
print("Batch size:", BATCH_SIZE)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    batch_dataset,
    batch_size=None,
    shuffle=True,
    num_workers=0,
)


# ============================================================
# MODEL
# ============================================================

model = nn.Sequential(
    nn.Linear(FEATURES, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)


# ============================================================
# LOSS
# ============================================================

loss_fn = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# TRAINING
# ============================================================

print()
print("========== TRAINING ==========")

total_start = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()

    epoch_start = time.perf_counter()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        # ------------------------------------------
        # Forward
        # ------------------------------------------

        logits = model(X_batch).squeeze(1)

        # ------------------------------------------
        # Loss
        # ------------------------------------------

        loss = loss_fn(logits, y_batch.float())

        # ------------------------------------------
        # Backprop
        # ------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        batch_samples = X_batch.shape[0]
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = total_loss / total_samples
    samples_per_second = total_samples / epoch_time

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Time: {epoch_time:.4f}s | "
        f"Samples/s: {samples_per_second:,.0f}"
    )

total_time = time.perf_counter() - total_start


# ============================================================
# CACHE
# ============================================================

print()
print("========== CACHE ==========")
print(dataset.cache_info())


# ============================================================
# RESULT
# ============================================================

print()
print("========== RESULT ==========")
print(f"Total training time: {total_time:.4f}s")
