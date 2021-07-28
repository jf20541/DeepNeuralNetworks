import torch 
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import HotelDataSet
from model import NeuralNerwork
import engine
import config


def train():
    # initiating custom Dataset and DataLoader
    dataset = HotelDataSet()
    dataloader = DataLoader(dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    
    # initiating model
    model = NeuralNerwork()
    # initiating Stochastic Gradient Descent to optimize parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    best_accuracy = 0
    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function 
        engine.train_fn(dataloader, model, optimizer)
        outputs, targets = engine.eval_fn(dataloader, model)
        outputs = np.array(outputs) >= 0.5
        # calculating accuracy score 
        accuracy = accuracy_score(targets, outputs)
        print(f'Epoch:{epochs+1}/{config.EPOCHS}, Accuracy Score = {accuracy:.4f})')
        if accuracy > best_accuracy:
            # save parameters into model.bin 
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
            
            
if __name__ == '__main__':
    train()