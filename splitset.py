import os
import numpy as np
import pandas as pd
import math
from data import dataManaging
import torch
from sklearn.model_selection import train_test_split

def splitData():

    d = pd.read_csv(os.path.join("train.csv"))

    x_data, y_data = dataManaging(d, False)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=5)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.05, random_state=5)

    x_train_id = x_train["PassengerId"].copy().to_numpy()
    x_train = x_train.drop("PassengerId", axis=1)

    x_val_id = x_val["PassengerId"].copy().to_numpy()
    x_val = x_val.drop("PassengerId", axis=1)

    x_test_id = x_test["PassengerId"].copy().to_numpy()
    x_test = x_test.drop("PassengerId", axis=1)

    x_train = torch.from_numpy(x_train.to_numpy())
    x_val = torch.from_numpy(x_val.to_numpy())
    x_test = torch.from_numpy(x_test.to_numpy())

    x_test_id = torch.from_numpy(x_test_id)
    x_train_id = torch.from_numpy(x_train_id)
    x_val_id = torch.from_numpy(x_val_id)

    y_train = torch.from_numpy(y_train.to_numpy())
    y_test = torch.from_numpy(y_test.to_numpy())
    y_val = torch.from_numpy(y_val.to_numpy())

    return x_train, y_train, x_train_id, x_val, y_val, x_val_id, x_test, y_test, x_test_id


