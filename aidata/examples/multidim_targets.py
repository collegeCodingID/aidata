"""
Multi-Dimensional Targets Example
==================================

This example shows how AIDATA handles targets that are NOT just 1D labels.

Use cases covered:
1. Segmentation masks        → y shape: (N, H, W)
2. Multi-label classification → y shape: (N, num_classes)
3. 3D volumetric targets     → y shape: (N, D, H, W)
"""

import numpy as np
import torch
from torch import nn

from aidata import AIDATAWriter, AIDATAReader, AIDATADataset, AIDATALoader


# ============================================================
# EXAMPLE 1: Semantic Segmentation
# ============================================================

print("=" * 70)
print("EXAMPLE 1: Semantic Segmentation")
print("=" * 70)

# Imagine: 1000 images, each 64x64 pixels, 10 classes
N, H, W, C = 1000, 64, 64, 10
num_features = 128

X = np.random.rand(N, num_features).astype(np.float32)
y = np.random.randint(0, C, size=(N, H, W), dtype=np.int64)

print(f"Data: X={X.shape}, y={y.shape}")

# Write
writer = AIDATAWriter("segmentation.aidata")
writer.write(X, y, metadata={
    "task": "semantic_segmentation",
    "num_classes": C,
    "image_size": [H, W],
}, chunk_size=256)

# Read
reader = AIDATAReader("segmentation.aidata")

print(f"Metadata: {reader.metadata['task']}")
print(f"y stored shape: {reader.metadata['y_shape']}")

# Single sample
x0, y0 = reader[0]
print(f"Sample 0: X={x0.shape}, y={y0.shape}")  # y=(64, 64)

# Batch
Xb, yb = reader.get_batch(0, 32)
print(f"Batch: X={Xb.shape}, y={yb.shape}")  # y=(32, 64, 64)

reader.close()


# ============================================================
# EXAMPLE 2: Multi-Label Classification
# ============================================================

print()
print("=" * 70)
print("EXAMPLE 2: Multi-Label Classification")
print("=" * 70)

# Imagine: Movie genres — each movie can have multiple genres
N, num_classes = 5000, 18

X = np.random.rand(N, 512).astype(np.float32)
y = np.random.randint(0, 2, size=(N, num_classes), dtype=np.int64)

print(f"Data: X={X.shape}, y={y.shape}")

writer = AIDATAWriter("multilabel.aidata")
writer.write(X, y, metadata={
    "task": "multilabel_classification",
    "num_classes": num_classes,
}, chunk_size=512)

reader = AIDATAReader("multilabel.aidata")

x0, y0 = reader[0]
print(f"Sample 0: X={x0.shape}, y={y0.shape}")  # y=(18,)
print(f"Labels for sample 0: {y0}")

reader.close()


# ============================================================
# EXAMPLE 3: 3D Volumetric Targets (Medical Imaging)
# ============================================================

print()
print("=" * 70)
print("EXAMPLE 3: 3D Volumetric Targets")
print("=" * 70)

# Imagine: CT scan volumes — each scan is 16x32x32
N, D, H, W = 200, 16, 32, 32

X = np.random.rand(N, 256).astype(np.float32)
y = np.random.rand(N, D, H, W).astype(np.float32)  # Continuous targets

print(f"Data: X={X.shape}, y={y.shape}")

writer = AIDATAWriter("volumetric.aidata")
writer.write(X, y, metadata={
    "task": "volumetric_regression",
    "volume_shape": [D, H, W],
}, chunk_size=64)

reader = AIDATAReader("volumetric.aidata")

x0, y0 = reader[0]
print(f"Sample 0: X={x0.shape}, y={y0.shape}")  # y=(16, 32, 32)

Xb, yb = reader.get_batch(0, 16)
print(f"Batch: X={Xb.shape}, y={yb.shape}")  # y=(16, 16, 32, 32)

reader.close()


# ============================================================
# EXAMPLE 4: PyTorch Training with Multi-Dim Targets
# ============================================================

print()
print("=" * 70)
print("EXAMPLE 4: PyTorch Training Loop")
print("=" * 70)

# Segmentation model
class SegmentationModel(nn.Module):
    def __init__(self, in_features, num_classes, h, w):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes * h * w),
        )
        self.num_classes = num_classes
        self.h = h
        self.w = w

    def forward(self, x):
        b = x.size(0)
        out = self.fc(x)
        return out.view(b, self.num_classes, self.h, self.w)


# Create small dataset
N = 256
X = np.random.rand(N, 128).astype(np.float32)
y = np.random.randint(0, 5, size=(N, 32, 32), dtype=np.int64)

writer = AIDATAWriter("train_seg.aidata")
writer.write(X, y, chunk_size=64, verbose=False)

# Native loader
loader = AIDATALoader(
    "train_seg.aidata",
    batch_size=32,
    shuffle=True,
)

model = SegmentationModel(128, 5, 32, 32)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training for 2 epochs...")
for epoch in range(2):
    total_loss = 0.0
    for X_batch, y_batch in loader:
        # X_batch: (32, 128)
        # y_batch: (32, 32, 32)
        
        logits = model(X_batch)  # (32, 5, 32, 32)
        loss = criterion(logits, y_batch.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss={total_loss / len(loader):.4f}")

loader.close()

print()
print("=" * 70)
print("✅ All multi-dimensional examples completed!")
print("=" * 70)
