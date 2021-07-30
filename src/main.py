import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from dataset import HotelDataSet2
from model import NeuralNerwork
import engine2
import config
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

"""
Training Set: x_train (features), x_test (targets)
Testing Set: y_train (features), y_test (targets)
"""
# print(x_train.shape) #     (82438, 31)
# print(y_train.shape) #     (20610, 31)

# print(x_test.shape) #      (82438,)
# print(y_test.shape) #      (20610,)

def train(fold, save_model=False):
    df = pd.read_csv(config.TRAINING_FOLDS)

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    ytrain = train_df.is_canceled.values
    xtrain = train_df.drop('is_canceled', axis=1).values

    yvalid = valid_df.is_canceled.values
    xvalid = valid_df.drop('is_canceled', axis=1).values
    
    print(ytrain.shape, xtrain.shape)
    print(yvalid.shape, xvalid.shape)

    
    

    # x_test = df_train["is_canceled"].values
    # y_test = df_test["is_canceled"].values

    # x_train = df_train.drop(["is_canceled"], axis=1).values
    # y_train = df_test.drop(["is_canceled"], axis=1).values

    # train_dataset = HotelDataSet2(x_train, y_train)
    # test_dataset = HotelDataSet2(x_test, y_test)

    # print(len(train_dataset), len(test_dataset))

    # train_loader = DataLoader(train_dataset, batch_size=512)
    # test_loader = DataLoader(test_dataset, batch_size=512)
    # # 323 iterations

    # model = NeuralNerwork()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # eng = engine2.Engine(model, optimizer)

    # best_loss = np.inf
    # early_stopping_iter = 10
    # early_stopping_counter = 0

    # for epoch in range(config.EPOCHS):
    #     train_loss = eng.train(train_loader)
    #     test_loss = eng.evaluate(test_loader)
    #     print(
    #         f"Fold:{fold}, Epoch:{epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}"
    #     )
    #     if test_loss <= best_loss:
    #         best_loss = test_loss
    #         if save_model:
    #             torch.save(model.state_dict(), f"../models/Mmodel{fold}.bin")
    #     else:
    #         early_stopping_counter += 1

    #     if early_stopping_counter > early_stopping_iter:
    #         break


if __name__ == "__main__":
    train(fold=0)
