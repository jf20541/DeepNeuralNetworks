import torch
from torch.utils.data import Dataset
import config
import pandas as pd
import numpy as np


class HotelDataSet:
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
            "features": torch.tensor(self.features[idx, :], dtype=torch.float),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }


"""
Training Set: x_train (features), x_test (targets)
Testing Set: y_train (features), y_test (targets)
"""
# print(ytrain.shape, xtrain.shape) (82438, 1) (82438, 31)
# print(yvalid.shape, xvalid.shape) (20610, 1) (20610, 31)
