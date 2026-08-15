import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aidata import (
    AIDATAWriter,
    AIDATAReader,
)


# ============================================================
# CONFIG
# ============================================================

SAMPLES = 100_000
FEATURES = 20
BATCH_SIZE = 256
EPOCHS = 3
SEED = 42

CSV_FILE = "benchmark_training.csv"
NPY_FILE = "benchmark_training.npy"
AIDATA_FILE = "benchmark_training.aidata"


# ============================================================
# HELPERS
# ============================================================

def file_size_mb(path):
    return Path(path).stat().st_size / (1024 * 1024)


def create_dataset():
    """Create ONE dataset. Every format uses exactly the same data."""
    rng = np.random.default_rng(SEED)

    X = rng.random((SAMPLES, FEATURES), dtype=np.float32)
    y = rng.integers(0, 2, size=SAMPLES, dtype=np.int64)

    return X, y


def make_model():
    """Create the same model for every benchmark."""
    return nn.Sequential(
        nn.Linear(FEATURES, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def get_initial_model_state():
    """Create one fixed initial model state."""
    torch.manual_seed(SEED)
    model = make_model()

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def train_model(batches, initial_state):
    """Train the exact same model using the supplied batches."""
    model = make_model()
    model.load_state_dict(initial_state)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_samples = 0
    final_loss = 0.0

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()

        epoch_samples = 0
        epoch_loss = 0.0

        for X_batch, y_batch in batches():
            logits = model(X_batch).squeeze(1)
            loss = loss_fn(logits, y_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_samples = X_batch.shape[0]
            epoch_samples += batch_samples
            epoch_loss += loss.item() * batch_samples

        total_samples += epoch_samples
        final_loss = epoch_loss / epoch_samples

    elapsed = time.perf_counter() - start
    processed_samples = SAMPLES * EPOCHS
    samples_per_second = processed_samples / elapsed

    return elapsed, samples_per_second, final_loss


# ============================================================
# CREATE DATA
# ============================================================

print("=" * 75)
print("AIDATA TRAINING BENCHMARK")
print("=" * 75)
print()
print(f"Dataset: {SAMPLES:,} samples × {FEATURES} features")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print()
print("Creating common dataset...")

X, y = create_dataset()


# ============================================================
# CREATE CSV
# ============================================================

print("Creating CSV...")

csv_start = time.perf_counter()

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    for i in range(SAMPLES):
        writer.writerow([*X[i], y[i]])

csv_write_time = time.perf_counter() - csv_start


# ============================================================
# CREATE NUMPY
# ============================================================

print("Creating NumPy...")

numpy_start = time.perf_counter()

numpy_data = np.empty((SAMPLES, FEATURES + 1), dtype=np.float32)
numpy_data[:, :FEATURES] = X
numpy_data[:, FEATURES] = y

np.save(NPY_FILE, numpy_data)

numpy_write_time = time.perf_counter() - numpy_start


# ============================================================
# CREATE AIDATA
# ============================================================

print("Creating AIDATA...")

aidata_start = time.perf_counter()

writer = AIDATAWriter(AIDATA_FILE)

writer.write(
    X,
    y,
    compression=True,
    chunk_size=4096,
)

aidata_write_time = time.perf_counter() - aidata_start


# ============================================================
# FILE SIZES
# ============================================================

csv_size = file_size_mb(CSV_FILE)
numpy_size = file_size_mb(NPY_FILE)
aidata_size = file_size_mb(AIDATA_FILE)


# ============================================================
# LOAD CSV
# ============================================================

print()
print("Loading CSV...")

csv_load_start = time.perf_counter()

csv_data = np.loadtxt(CSV_FILE, delimiter=",", dtype=np.float32)
csv_X = csv_data[:, :FEATURES]
csv_y = csv_data[:, FEATURES].astype(np.int64)

csv_load_time = time.perf_counter() - csv_load_start


# ============================================================
# LOAD NUMPY
# ============================================================

print("Loading NumPy...")

numpy_load_start = time.perf_counter()

numpy_loaded = np.load(NPY_FILE)
numpy_X = numpy_loaded[:, :FEATURES]
numpy_y = numpy_loaded[:, FEATURES].astype(np.int64)

numpy_load_time = time.perf_counter() - numpy_load_start


# ============================================================
# OPEN AIDATA
# ============================================================

print("Opening AIDATA...")

aidata_load_start = time.perf_counter()

aidata = AIDATAReader(AIDATA_FILE, cache_size=8)

aidata_load_time = time.perf_counter() - aidata_load_start


# ============================================================
# FIXED MODEL INITIALIZATION
# ============================================================

initial_state = get_initial_model_state()


# ============================================================
# CSV BATCH GENERATOR
# ============================================================

def csv_batches():
    for start in range(0, SAMPLES, BATCH_SIZE):
        end = min(start + BATCH_SIZE, SAMPLES)

        X_batch = torch.from_numpy(csv_X[start:end])
        y_batch = torch.from_numpy(csv_y[start:end])

        yield (X_batch, y_batch)


# ============================================================
# NUMPY BATCH GENERATOR
# ============================================================

def numpy_batches():
    for start in range(0, SAMPLES, BATCH_SIZE):
        end = min(start + BATCH_SIZE, SAMPLES)

        X_batch = torch.from_numpy(numpy_X[start:end])
        y_batch = torch.from_numpy(numpy_y[start:end])

        yield (X_batch, y_batch)


# ============================================================
# AIDATA BATCH GENERATOR
# ============================================================

def aidata_batches():
    for start in range(0, SAMPLES, BATCH_SIZE):
        X_batch, y_batch = aidata.get_batch(
            start=start,
            batch_size=BATCH_SIZE,
        )

        X_batch = torch.from_numpy(X_batch)
        y_batch = torch.from_numpy(y_batch)

        yield (X_batch, y_batch)


# ============================================================
# WARMUP
# ============================================================

print()
print("Running warmup...")

for X_batch, y_batch in csv_batches():
    break

for X_batch, y_batch in numpy_batches():
    break

for X_batch, y_batch in aidata_batches():
    break


# ============================================================
# TRAIN CSV
# ============================================================

print()
print("Training CSV...")

csv_train_time, csv_samples_sec, csv_loss = train_model(
    csv_batches,
    initial_state,
)


# ============================================================
# TRAIN NUMPY
# ============================================================

print("Training NumPy...")

numpy_train_time, numpy_samples_sec, numpy_loss = train_model(
    numpy_batches,
    initial_state,
)


# ============================================================
# CLEAR AIDATA CACHE
# ============================================================

aidata.clear_cache()


# ============================================================
# TRAIN AIDATA
# ============================================================

print("Training AIDATA...")

aidata_train_time, aidata_samples_sec, aidata_loss = train_model(
    aidata_batches,
    initial_state,
)


# ============================================================
# TOTAL TIMES
# ============================================================

csv_total = csv_load_time + csv_train_time
numpy_total = numpy_load_time + numpy_train_time
aidata_total = aidata_load_time + aidata_train_time


# ============================================================
# CACHE
# ============================================================

cache_info = aidata.cache_info()


# ============================================================
# RESULTS
# ============================================================

print()
print()
print("=" * 110)
print("FINAL COMPARISON")
print("=" * 110)
print()

print(
    f"{'Metric':<22}"
    f"{'CSV':>18}"
    f"{'NumPy':>18}"
    f"{'AIDATA':>18}"
)

print("-" * 110)

print(
    f"{'File size (MB)':<22}"
    f"{csv_size:>18.2f}"
    f"{numpy_size:>18.2f}"
    f"{aidata_size:>18.2f}"
)

print(
    f"{'Write time (s)':<22}"
    f"{csv_write_time:>18.4f}"
    f"{numpy_write_time:>18.4f}"
    f"{aidata_write_time:>18.4f}"
)

print(
    f"{'Load/Open time (s)':<22}"
    f"{csv_load_time:>18.4f}"
    f"{numpy_load_time:>18.4f}"
    f"{aidata_load_time:>18.4f}"
)

print(
    f"{'Training time (s)':<22}"
    f"{csv_train_time:>18.4f}"
    f"{numpy_train_time:>18.4f}"
    f"{aidata_train_time:>18.4f}"
)

print(
    f"{'Total time (s)':<22}"
    f"{csv_total:>18.4f}"
    f"{numpy_total:>18.4f}"
    f"{aidata_total:>18.4f}"
)

print(
    f"{'Samples/sec':<22}"
    f"{csv_samples_sec:>18,.0f}"
    f"{numpy_samples_sec:>18,.0f}"
    f"{aidata_samples_sec:>18,.0f}"
)

print(
    f"{'Final loss':<22}"
    f"{csv_loss:>18.4f}"
    f"{numpy_loss:>18.4f}"
    f"{aidata_loss:>18.4f}"
)

print("-" * 110)

print()
print("AIDATA cache:")
print(cache_info)

print()
print("=" * 110)


# ============================================================
# BEST RESULTS
# ============================================================

training_times = {
    "CSV": csv_train_time,
    "NumPy": numpy_train_time,
    "AIDATA": aidata_train_time,
}

total_times = {
    "CSV": csv_total,
    "NumPy": numpy_total,
    "AIDATA": aidata_total,
}

smallest_file = min(
    {"CSV": csv_size, "NumPy": numpy_size, "AIDATA": aidata_size},
    key={"CSV": csv_size, "NumPy": numpy_size, "AIDATA": aidata_size}.get,
)

fastest_training = min(training_times, key=training_times.get)
fastest_total = min(total_times, key=total_times.get)

print()
print("========== WINNERS ==========")
print(f"Smallest file:      {smallest_file}")
print(f"Fastest training:   {fastest_training}")
print(f"Fastest total:      {fastest_total}")


# ============================================================
# CLEANUP
# ============================================================

print()
print("Cleaning temporary files...")

for path in [CSV_FILE, NPY_FILE, AIDATA_FILE]:
    if os.path.exists(path):
        os.remove(path)

print("Done.")
