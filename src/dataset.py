import torch
from torch.utils.data import Dataset
import config
import pandas as pd
import numpy as np

# inheret function from dataset class torch
class HotelDataSet(Dataset):
    # data loading
    def __init__(self):
        xy = pd.read_csv(config.TRAINING_FILE).values
        xy = xy.astype(np.float32)
        self.x = torch.from_numpy(xy[:, :-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __len__(self):
        return self.len

    # allow indexing later
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
