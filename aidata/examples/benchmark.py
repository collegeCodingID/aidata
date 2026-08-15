import os
import time
from pathlib import Path

import numpy as np

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
CHUNK_SIZES = [256, 1024, 4096, 16384]
RUNS = 5
AIDATA_FILE = "benchmark.aidata"
NPY_FILE = "benchmark.npy"


# ============================================================
# DATA
# ============================================================

np.random.seed(42)

print("Creating dataset...")

X = np.random.rand(SAMPLES, FEATURES).astype(np.float32)
y = np.random.randint(0, 2, size=SAMPLES, dtype=np.int64)


# ============================================================
# HELPERS
# ============================================================

def mb(size):
    return size / (1024 * 1024)


def file_size(path):
    return Path(path).stat().st_size


def benchmark(function, runs=5):
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.mean(times))


# ============================================================
# NUMPY
# ============================================================

print("\nCreating NumPy file...")

np.save(
    NPY_FILE,
    np.column_stack([X, y])
)

numpy_size = file_size(NPY_FILE)
numpy_read_time = benchmark(lambda: np.load(NPY_FILE), RUNS)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# AIDATA
# ============================================================

for chunk_size in CHUNK_SIZES:
    print(f"\nTesting chunk size: {chunk_size}")

    # ------------------------------------------
    # Delete old file
    # ------------------------------------------

    if os.path.exists(AIDATA_FILE):
        os.remove(AIDATA_FILE)

    # ------------------------------------------
    # Write
    # ------------------------------------------

    writer = AIDATAWriter(AIDATA_FILE)

    write_start = time.perf_counter()

    writer.write(
        X,
        y,
        compression=True,
        chunk_size=chunk_size,
    )

    write_time = time.perf_counter() - write_start

    aidata_size = file_size(AIDATA_FILE)

    # ------------------------------------------
    # Reader
    # ------------------------------------------

    dataset = AIDATAReader(AIDATA_FILE, cache_size=8)

    # ------------------------------------------
    # Open
    # ------------------------------------------

    open_time = benchmark(
        lambda: AIDATAReader(AIDATA_FILE, cache_size=8),
        RUNS
    )

    # ------------------------------------------
    # Random sample
    # ------------------------------------------

    random_index = SAMPLES // 2

    dataset.clear_cache()

    random_time = benchmark(
        lambda: dataset[random_index],
        RUNS
    )

    # ------------------------------------------
    # Random batch
    # ------------------------------------------

    dataset.clear_cache()

    random_batch_time = benchmark(
        lambda: dataset.get_batch(random_index, BATCH_SIZE),
        RUNS
    )

    # ------------------------------------------
    # Sequential
    # ------------------------------------------

    def sequential_read():
        dataset.clear_cache()

        for start in range(0, SAMPLES, BATCH_SIZE):
            dataset.get_batch(start, BATCH_SIZE)

    sequential_time = benchmark(sequential_read, 1)

    # ------------------------------------------
    # Full chunks
    # ------------------------------------------

    def full_read():
        dataset.clear_cache()

        for chunk_id in range(dataset.chunk_count):
            dataset.get_chunk(chunk_id)

    full_time = benchmark(full_read, 1)

    # ------------------------------------------
    # Cache
    # ------------------------------------------

    cache = dataset.cache_info()

    # ------------------------------------------
    # Save
    # ------------------------------------------

    results.append({
        "chunk_size": chunk_size,
        "chunks": dataset.chunk_count,
        "size_mb": mb(aidata_size),
        "write": write_time,
        "open": open_time,
        "random": random_time,
        "batch": random_batch_time,
        "sequential": sequential_time,
        "full": full_time,
        "cache_hits": cache["cache_hits"],
        "cache_misses": cache["cache_misses"],
    })


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 125)
print("                         AIDATA BENCHMARK")
print("=" * 125)
print(f"Dataset: {SAMPLES:,} samples × {FEATURES} features")
print()
print(f"NumPy size: {mb(numpy_size):.2f} MB")
print(f"NumPy full read: {numpy_read_time:.6f}s")
print()
print(
    "Chunk | Chunks | Size | Write | Open | Random | Batch | Sequential | Full"
)
print("-" * 125)

for result in results:
    print(
        f"{result['chunk_size']:5d} | "
        f"{result['chunks']:6d} | "
        f"{result['size_mb']:5.2f} | "
        f"{result['write']:.4f} | "
        f"{result['open']:.6f} | "
        f"{result['random']:.6f} | "
        f"{result['batch']:.6f} | "
        f"{result['sequential']:.4f} | "
        f"{result['full']:.4f}"
    )

print("=" * 125)


# ============================================================
# CLEANUP
# ============================================================

for file in [AIDATA_FILE, NPY_FILE]:
    if os.path.exists(file):
        os.remove(file)
