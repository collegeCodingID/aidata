import math

import torch
from torch.utils.data import Dataset


class AIDATAPyTorchDataset(Dataset):
    """Sample-level PyTorch Dataset that wraps an AIDATA reader.

    Parameters
    ----------
    dataset : AIDATAReader
        An open reader instance.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]

        X = torch.from_numpy(X.copy())
        y = torch.tensor(y)

        return X, y


class AIDATABatchDataset(Dataset):
    """Batch-level PyTorch Dataset that wraps an AIDATA reader.

    One Dataset item = one complete batch.

    Parameters
    ----------
    dataset : AIDATAReader
        An open reader instance.
    batch_size : int
        Number of samples per batch.
    """

    def __init__(self, dataset, batch_size=256):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(dataset) / batch_size)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_index):
        if batch_index < 0:
            batch_index += self.num_batches

        if batch_index < 0 or batch_index >= self.num_batches:
            raise IndexError("Batch index out of range")

        start = batch_index * self.batch_size

        X, y = self.dataset.get_batch(
            start=start,
            batch_size=self.batch_size,
        )

        X = torch.from_numpy(X.copy())
        y = torch.from_numpy(y.copy())

        return X, y
