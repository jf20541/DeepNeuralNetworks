import torch
import config
from dataset import HotelDataSet
from model import NeuralNerwork
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.autograd import Variable
import math
import engine
import numpy as np


def train():
    dataset = HotelDataSet()
    dataloader = DataLoader(
        dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )

    model = NeuralNerwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    best_accuracy = 0
    for epochs in range(config.EPOCHS):
        engine.train_fn(dataloader, model, optimizer)
        outputs, targets = engine.eval_fn(dataloader, model)
        outputs = np.array(outputs) >= 0.5
        accuracy = accuracy_score(targets, outputs)
        print(f"Epoch:{epochs+1}/{config.EPOCHS}, Accuracy Score = {accuracy:.4f}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    train()
