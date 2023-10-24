import os
import numpy as np
import pandas as pd
import math
from data import dataManaging
import torch
from splitset import splitData
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn


class customSet(Dataset):

    def __init__(self, tensor, transform=None):
        self.tensor = tensor
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensor[0][index]

        if self.transform:
            x = self.transform(x)
        y = self.tensor[1][index]
        return x, y

    def __len__(self):
        return self.tensor[0].size(0)


class network(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(76, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.double()
    def forward(self, x):
        return (self.stack(x))


x_train, y_train, x_train_id, x_val, y_val, x_val_id, x_test, y_test, x_test_id = splitData()


train = customSet(tensor=[x_train, y_train])
val = customSet(tensor=[x_val, y_val])
test = customSet(tensor=[x_test, y_test])

device = "cpu"

train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val, batch_size=65, shuffle=True)
test_dataloader = DataLoader(test, batch_size=64, shuffle=True)

model = network()

print((model(x_train[0])))

