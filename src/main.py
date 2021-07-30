import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from dataset import HotelDataSet2
from model import NeuralNerwork
import engine
import config
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

"""
Training Set: x_train (features), x_test (targets)
Testing Set: y_train (features), y_test (targets)
"""


def train(fold):
    df = pd.read_csv(config.TRAINING_FOLDS)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    x_test = df_train["is_canceled"].values
    y_test = df_test["is_canceled"].values

    x_train = df_train.drop(["is_canceled", "kfold"], axis=1).values
    y_train = df_test.drop(["is_canceled", "kfold"], axis=1).values

    train_dataset = HotelDataSet2(x_train, y_train)
    test_dataset = HotelDataSet2(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    model = NeuralNerwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_accuracy = 0
    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        engine.train_fn(train_loader, model, optimizer)
        outputs, targets = engine.eval_fn(test_loader, model)
        outputs = np.array(outputs) >= 0.5
        # calculating accuracy score
        roc_auc = roc_auc_score(targets, outputs)
        print(f"Epoch:{epochs+1}/{config.EPOCHS}, ROC AUC:{roc_auc:.4f}")
        if roc_auc > best_accuracy:
            # save parameters into model.bin
            torch.save(model.state_dict(), config.MODEL_PATH)


if __name__ == "__main__":
    train(fold=1)
