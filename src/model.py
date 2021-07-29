import torch.nn as nn
import torch.nn.functional as F


class NeuralNerwork(nn.Module):
    def __init__(self):
        super(NeuralNerwork, self).__init__()
        self.fc1 = nn.Linear(30, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


# model = NeuralNerwork()
# print(model)
