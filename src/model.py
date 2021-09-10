import torch.nn as nn
import torch.nn.functional as F


class NeuralNerwork(nn.Module):
    def __init__(self, n_features, n_targets):
        super(NeuralNerwork, self).__init__()
        # input layer with n_features (nodes) and 15 nodes for 2nd layer
        self.fc1 = nn.Linear(n_features, 15)
        # 15 nodes and next hidden layer 1ith 10 nodes
        self.fc2 = nn.Linear(15, 10)
        # 10 nodes and next n_targets nodes
        self.fc3 = nn.Linear(10, n_targets)
        # add drop out to avoid overfitting on 2 hidden layers
        self.dropout = nn.Dropout(0.20)
        # sigmoid function as last yaer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))
