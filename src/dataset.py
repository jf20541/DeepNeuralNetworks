import torch
from torch.utils.data import Dataset
import config
import pandas as pd
import numpy as np


# # inheret function from dataset class torch
class HotelDataSet(Dataset):
    # data loading
    def __init__(self):
        xy = pd.read_csv(config.TRAINING_FILE).values
        xy = xy.astype(np.float32)
        self.x = torch.from_numpy(xy[:, :-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    # return data length
    def __len__(self):
        return self.len

    # return item on the index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class HotelDataSet2(Dataset):
    # data loading
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    # return data length
    def __len__(self):
        return self.features.shape[0]

    # return item on the index
    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.features[idx], dtype=torch.float),
            "y": torch.tensor(self.targets[idx], dtype=torch.float),
        }
