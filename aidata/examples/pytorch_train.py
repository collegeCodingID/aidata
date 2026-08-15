import time

import torch
from torch import nn

from aidata import AIDATALoader


# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = "training.aidata"
BATCH_SIZE = 256
EPOCHS = 3
LEARNING_RATE = 0.001
SEED = 42


# ============================================================
# SETUP
# ============================================================

torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("AIDATA NATIVE PYTORCH LOADER")
print("=" * 70)
print()
print(f"Device: {device}")


# ============================================================
# LOADER
# ============================================================

loader = AIDATALoader(
    path=DATASET_FILE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache_size=8,
    drop_last=False,
    device=device,
    seed=SEED,
)

print(f"Samples: {loader.num_samples:,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches/epoch: {len(loader):,}")


# ============================================================
# TEST FIRST BATCH
# ============================================================

first_X = None
first_y = None

for X, y in loader:
    first_X = X
    first_y = y
    break

print()
print("First batch:")
print(f"X shape: {tuple(first_X.shape)}")
print(f"X dtype: {first_X.dtype}")
print(f"Y shape: {tuple(first_y.shape)}")
print(f"Y dtype: {first_y.dtype}")


# ============================================================
# MODEL
# ============================================================

input_features = first_X.shape[1]

model = nn.Sequential(
    nn.Linear(input_features, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)


# ============================================================
# LOSS
# ============================================================

loss_fn = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# TRAINING
# ============================================================

print()
print("Training...")

training_start = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        y_batch = y_batch.float()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(X_batch).squeeze(1)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = loss_fn(logits, y_batch)

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        current_batch_size = X_batch.shape[0]
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    epoch_loss = total_loss / total_samples

    print(f"{epoch + 1}/{EPOCHS} loss: {epoch_loss:.6f}")

training_time = time.perf_counter() - training_start


# ============================================================
# PERFORMANCE
# ============================================================

processed_samples = loader.num_samples * EPOCHS
samples_per_second = processed_samples / training_time


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print()
print(f"Training time: {training_time:.4f}s")
print(f"Samples/sec: {samples_per_second:,.0f}")


# ============================================================
# CACHE
# ============================================================

print()
print("Cache:")
print(loader.cache_info())


# ============================================================
# CLEANUP
# ============================================================

loader.close()

print()
print("Training complete.")
