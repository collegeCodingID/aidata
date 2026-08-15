import time

import torch
from torch import nn

from aidata import AIDATALoader


DATASET = "training.aidata"
BATCH_SIZE = 256
EPOCHS = 3


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("AIDATA PIPELINE PROFILER")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# LOADER
# ============================================================

loader = AIDATALoader(
    path=DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache_size=8,
    device=device,
)


print(f"Samples: {loader.num_samples:,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches: {len(loader):,}")


# ============================================================
# GET FIRST BATCH
# ============================================================

X, y = next(iter(loader))

input_features = X.shape[1]


print()
print("First batch:")
print("X:", X.shape)
print("Y:", y.shape)


# ============================================================
# MODEL
# ============================================================

model = nn.Sequential(
    nn.Linear(input_features, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)


loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# ============================================================
# PROFILE VARIABLES
# ============================================================

read_time = 0.0
compute_time = 0.0

total_samples = 0


# ============================================================
# MANUAL TRAINING LOOP
# ============================================================

print()
print("Profiling...")


total_start = time.perf_counter()


for epoch in range(EPOCHS):

    model.train()

    epoch_read = 0.0
    epoch_compute = 0.0

    epoch_samples = 0

    iterator = iter(loader)

    while True:

        # ----------------------------------------------------
        # AIDATA READ
        # ----------------------------------------------------

        read_start = time.perf_counter()

        try:
            X_batch, y_batch = next(iterator)

        except StopIteration:
            break

        read_end = time.perf_counter()

        batch_read_time = (
            read_end - read_start
        )

        epoch_read += batch_read_time

        # ----------------------------------------------------
        # PYTORCH COMPUTATION
        # ----------------------------------------------------

        compute_start = time.perf_counter()

        y_batch = y_batch.float()

        logits = model(
            X_batch
        ).squeeze(1)

        loss = loss_fn(
            logits,
            y_batch,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        compute_end = time.perf_counter()

        batch_compute_time = (
            compute_end
            - compute_start
        )

        epoch_compute += (
            batch_compute_time
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        batch_samples = X_batch.shape[0]

        epoch_samples += batch_samples


    read_time += epoch_read

    compute_time += epoch_compute

    total_samples += epoch_samples

    print(
        f"Epoch {epoch + 1}: "
        f"read={epoch_read:.4f}s | "
        f"compute={epoch_compute:.4f}s"
    )


total_time = (
    time.perf_counter()
    - total_start
)


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 70)
print("PROFILE RESULT")
print("=" * 70)

print()

print(
    f"Total time:       {total_time:.4f}s"
)

print(
    f"AIDATA read time:  {read_time:.4f}s"
)

print(
    f"PyTorch compute:   {compute_time:.4f}s"
)

print(
    f"Other overhead:    "
    f"{total_time - read_time - compute_time:.4f}s"
)

print()

print(
    f"Samples/sec: "
    f"{total_samples / total_time:,.0f}"
)

print()

print("Cache:")

print(
    loader.cache_info()
)


loader.close()
