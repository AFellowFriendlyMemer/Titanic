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
import torch.optim
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
            nn.Linear(40, 2),
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

train_dataloader = DataLoader(train, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test, batch_size=1, shuffle=True)

model = network()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)



def trainEpoch():

    running_loss = 0
    last_loss = 0
    prev_param = list(model.parameters())
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs.requires_grad_()
        optimizer.zero_grad()

        pred = model(inputs)
        loss = loss_fn(pred, labels)
        loss.backward()

        optimizer.step()

        # print(list(model.parameters()) == prev_param)
        prev_param = list(model.parameters())
        running_loss += loss.item()

        if i % 50 == 49:
            last_loss = running_loss / 50
            print(f"Batch {i + 1} Loss: {last_loss}")
        running_loss = 0
        
    return last_loss

epoch_num = 0
epochs = 10
best_vloss = 1000000
for epoch in range(epochs):
    print(f"Epoch {epoch_num+1}: ")

    model.train(True)

    avg_loss = trainEpoch()
    running_vloss = 0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            loss = loss_fn(voutputs, vlabels)
            running_vloss += loss
    
    avg_vloss = running_vloss / (i+ 1)
    print(f"Train {avg_loss}   Valid {avg_vloss}")
    epoch_num += 1
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_03072007_{}".format(epoch_num)
        torch.save(model.state_dict(), model_path)






