"""Dataset utilities for pairwise training."""

from torch.utils.data import Dataset


class PairwiseDataset(Dataset):
    """Pairwise dataset for (Xi, Xj, y)."""

    def __init__(self, Xi, Xj, y):
        self.Xi = Xi
        self.Xj = Xj
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.Xi[idx], self.Xj[idx], self.y[idx]
