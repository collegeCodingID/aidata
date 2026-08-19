import numpy as np

from aidata import (
    AIDATAWriter,
    AIDATAReader,
)

from aidata.integrations import (
    AIDATAPyTorchDataset,
    AIDATABatchDataset,
)


# ============================================================
# CREATE DATA
# ============================================================

print("Creating dataset...")

X = np.random.rand(10000, 20).astype(np.float32)
y = np.random.randint(0, 2, size=10000, dtype=np.int64)


# ============================================================
# WRITE
# ============================================================

print("\nWriting AIDATA...")

writer = AIDATAWriter("training.aidata")

writer.write(
    X,
    y,
    metadata={
        "dataset_name": "Binary Classification",
        "description": "AIDATA V0.5 test dataset",
        "task": "binary_classification",
    },
    compression=True,
    chunk_size=4096,
)


# ============================================================
# READ
# ============================================================

print("\nOpening AIDATA...")

dataset = AIDATAReader("training.aidata", cache_size=4)


# ============================================================
# INFO
# ============================================================

print("\n========== INFO ==========")

print("Samples:", len(dataset))
print("Features:", dataset.metadata["features"])
print("Chunk size:", dataset.metadata["chunk_size"])
print("Chunks:", dataset.chunk_count)
print("Compression:", dataset.metadata["compression"])


# ============================================================
# SINGLE SAMPLE
# ============================================================

print("\n========== SAMPLE ==========")

X_sample, y_sample = dataset[5000]

print("X shape:", X_sample.shape)
print("Y:", y_sample)


# ============================================================
# BATCH
# ============================================================

print("\n========== BATCH ==========")

X_batch, y_batch = dataset.get_batch(start=5000, batch_size=256)

print("X batch:", X_batch.shape)
print("Y batch:", y_batch.shape)


# ============================================================
# CACHE
# ============================================================

print("\n========== CACHE TEST ==========")

print("Initial:", dataset.cache_info())

dataset.get_chunk(0)

print("After chunk 0:", dataset.cache_info())

# Read chunk 0 again — this should produce a cache HIT.
dataset.get_chunk(0)

print("After chunk 0 again:", dataset.cache_info())

# Read the remaining valid chunks.
# For 10,000 samples and chunk_size=4096:
#   chunk 0 -> samples 0-4095
#   chunk 1 -> samples 4096-8191
#   chunk 2 -> samples 8192-9999

for chunk_id in range(dataset.chunk_count):
    dataset.get_chunk(chunk_id)

print("After reading all chunks:", dataset.cache_info())


# ============================================================
# PYTORCH SAMPLE DATASET
# ============================================================

print("\n========== PYTORCH SAMPLE DATASET ==========")

torch_dataset = AIDATAPyTorchDataset("training.aidata")

X_torch, y_torch = torch_dataset[0]

print("X:", X_torch.shape)
print("Y:", y_torch.shape)
print("X dtype:", X_torch.dtype)
print("Y dtype:", y_torch.dtype)


# ============================================================
# PYTORCH BATCH DATASET
# ============================================================

print("\n========== PYTORCH BATCH DATASET ==========")

#batch_dataset = AIDATABatchDataset(dataset, batch_size=256)
batch_dataset = AIDATABatchDataset("training.aidata", batch_size=256)


X_torch, y_torch = batch_dataset[0]

print("X batch:", X_torch.shape)
print("Y batch:", y_torch.shape)

print("\nDone.")
