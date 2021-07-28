import torch.nn as nn

class NeuralNerwork(nn.Module):
    def __init__(self):
        super(NeuralNerwork, self).__init__()
        self.fc1 = nn.Linear(30, 10)
        self.fc2 = nn.Linear(10,8)
        self.fc3 = nn.Linear(8,4)
        self.fc4 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.sigmoid(self.fc4(x))