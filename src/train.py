import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from dataset import HotelDataSet
from model import NeuralNerwork
from engine import Engine
import config


def train():
    df = pd.read_csv(config.TRAINING_FILE)

    # define target and features values as numpy arrays
    targets = df[["is_canceled"]].values
    features = df.drop("is_canceled", axis=1).values

    # split data, trainingset 80% and testingset 20%, stratify targets values bc its skewed
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, stratify=targets
    )

    # initiate custom dataset and feed to dataloader
    train_dataset = HotelDataSet(features=x_train, targets=y_train)
    test_dataset = HotelDataSet(features=x_test, targets=y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    # initiate NeuralNerwork, Adam optimizer, and Engine class
    model = NeuralNerwork(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    eng = Engine(model, optimizer)

    best_accuracy = np.inf
    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        train_targets, train_outputs = eng.train_fn(train_loader)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)
        eval_outputs = np.array(eval_outputs) >= 0.5
        # calculating accuracy score
        train_metric = roc_auc_score(train_targets, train_outputs)
        eval_metric = roc_auc_score(eval_targets, eval_outputs)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Train ROC-AUC: {train_metric:.4f}, Eval ROC-AUC: {eval_metric:.4f}"
        )
        if eval_metric > best_accuracy:
            # save parameters into model.bin
            torch.save(model.state_dict(), config.MODEL_PATH)


if __name__ == "__main__":
    train()
