import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader


# ============================================================
# AIDATA import
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidata import AIDATAReader


# ============================================================
# Configuration
# ============================================================

AIDATA_FILE = ROOT / "training.aidata"
BATCH_SIZE = 256
EPOCHS = 3
NUM_FEATURES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# AIDATA -> PyTorch Dataset
# ============================================================

class AIDATADataset(IterableDataset):

    def __init__(self, file_path, batch_size=256):
        super().__init__()

        self.file_path = str(file_path)
        self.batch_size = batch_size

    def _normalize_chunk(self, chunk):
        """
        Convert different possible AIDATA chunk formats into:
            X -> torch.float32
            y -> torch.int64
        """

        # ----------------------------------------------------
        # Tuple/list: (X, y)
        # ----------------------------------------------------

        if isinstance(chunk, (tuple, list)) and len(chunk) >= 2:
            X = chunk[0]
            y = chunk[1]

        # ----------------------------------------------------
        # Dictionary: {"X": ..., "y": ...}
        # ----------------------------------------------------

        elif isinstance(chunk, dict):
            if "X" in chunk:
                X = chunk["X"]
            elif "x" in chunk:
                X = chunk["x"]
            elif "features" in chunk:
                X = chunk["features"]
            else:
                raise ValueError(
                    "Could not find X/features in AIDATA chunk"
                )

            if "y" in chunk:
                y = chunk["y"]
            elif "Y" in chunk:
                y = chunk["Y"]
            elif "target" in chunk:
                y = chunk["target"]
            elif "label" in chunk:
                y = chunk["label"]
            else:
                raise ValueError(
                    "Could not find y/target/label in AIDATA chunk"
                )

        else:
            raise TypeError(
                f"Unsupported AIDATA chunk type: {type(chunk)}"
            )

        # ----------------------------------------------------
        # NumPy/list -> Tensor
        # ----------------------------------------------------

        X = torch.from_numpy(
            np.array(X, dtype=np.float32, copy=True)
        )

        y = torch.from_numpy(
            np.array(y, dtype=np.int64, copy=True)
        )

        return X, y

    def __iter__(self):
        reader = AIDATAReader(self.file_path)

        chunk_index = 0

        while True:
            try:
                chunk = reader.get_chunk(chunk_index)

            except IndexError:
                break

            X, y = self._normalize_chunk(chunk)

            yield X, y

            chunk_index += 1


# ============================================================
# Model
# ============================================================

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# Create Dataset
# ============================================================

dataset = AIDATADataset(
    AIDATA_FILE,
    batch_size=BATCH_SIZE
)


# ============================================================
# PyTorch DataLoader
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# Model
# ============================================================

model = Model().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# Information
# ============================================================

print("=" * 60)
print("AIDATA + PyTorch DataLoader")
print("=" * 60)
print()
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Features: {NUM_FEATURES}")
print(f"Epochs: {EPOCHS}")
print()


# ============================================================
# Check first batch
# ============================================================

first_batch = next(iter(loader))

X, y = first_batch

print("First batch:")
print(f"X shape: {tuple(X.shape)}")
print(f"X dtype: {X.dtype}")
print(f"Y shape: {tuple(y.shape)}")
print(f"Y dtype: {y.dtype}")
print()


# ============================================================
# Training
# ============================================================

print("Training...")

start_time = time.perf_counter()

total_samples = 0

for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0.0
    epoch_samples = 0

    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # Forward
        logits = model(X)

        # Loss
        loss = criterion(logits, y.float())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_samples += batch_size

    avg_loss = epoch_loss / epoch_samples
    total_samples += epoch_samples

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {avg_loss:.6f}"
    )

end_time = time.perf_counter()
training_time = end_time - start_time


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("RESULT")
print("=" * 60)
print()
print(f"Training time: {training_time:.4f}s")
print(f"Samples processed: {total_samples:,}")
print(f"Samples/sec: {total_samples / training_time:,.0f}")
print()
print("Training complete.")
