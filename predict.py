import os
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.optim
from sklearn.model_selection import train_test_split

def dataManaging(train, isTest):

    cabin_num = []



    for i in range(len(train)):
        if type(train.iloc[i]["Cabin"]) == float:
            cabin_num.append(0)
        else:
            cabin_num.append(len(train.iloc[i]["Cabin"].split(" ")))

    train["Cabin_Num"] = cabin_num

    cabin_letter = []

    for i in range(len(train)):
        if type(train.iloc[i]["Cabin"]) == float:
            cabin_letter.append(0)
        else:
            cabin_letter.append(train.iloc[i]["Cabin"][0])

    train["Cabin_Letter"] = cabin_letter


    c_indexes = [0, 'C', 'B', 'D', 'E', 'A', 'F', 'G', 'T']

    for i in range(1, len(c_indexes)):
        train["Cabin_" + c_indexes[i]] = (train["Cabin_Letter"] == c_indexes[i]).astype(int)

    train = train.drop("Cabin_Letter", axis=1)
    train = train.drop("Cabin", axis=1)
    age_median = 28.0

    for i in range(len(train)):
        if math.isnan(train.iloc[i]["Age"]):
            train.at[i, "Age"] = age_median

    train["Sex"] = (train["Sex"] == "male").astype(int)

    ticket_f = []

    for i in range(len(train)):
        spl = train.at[i, "Ticket"].split(" ")
        if len(spl) > 1:
            ticket_f.append(spl[0])
        else:
            ticket_f.append(0)


    train["Ticket_f"] = ticket_f

    c_ticket = [           0,         'PC',       'C.A.',     'STON/O',        'A/5',
                'W./C.',        'CA.', 'SOTON/O.Q.',   'SOTON/OQ',       'A/5.',
                'CA',   'STON/O2.',          'C',     'F.C.C.',     'S.O.C.',
            'SC/PARIS',   'SC/Paris',  'S.O./P.P.',         'PP',       'A/4.',
                'A/4',      'SC/AH',      'A./5.',   'SOTON/O2',       'A.5.',
                'WE/P', 'S.C./PARIS',       'P/PP',       'F.C.',         'SC',
            'S.W./PP',        'A/S',         'Fa',      'SCO/W',      'SW/PP',
                'W/C',  'S.C./A.4.',     'S.O.P.',        'A4.',     'W.E.P.',
                'SO/C',       'S.P.', 'C.A./SOTON']
    for i in range(1, len(c_ticket)):
        train["Ticket_f_" + c_ticket[i]] = (train["Ticket_f"] == c_ticket[i]).astype(int)

    train = train.drop("Ticket", axis=1)

    for i in range(len(train)):
        n = train.at[i, "Name"]
        n = n[n.find(','):n.find('.')][2:]
        train.at[i, "Name"] = n

    c_name = ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Mlle', 'Major', 'Col',
        'the Countess', 'Capt', 'Ms', 'Sir', 'Lady', 'Mme', 'Don', 'Jonkheer']
    for i in range(1, len(c_name)):
        train["Name_" + c_name[i]] = (train["Name"] == c_name[i]).astype(int)

    train = train.drop("Name", axis=1)

    for i in range(0, len(train["Embarked"].value_counts().index)):
        train["Embarked_" + train["Embarked"].value_counts().index[i]] = (train["Embarked"] == train["Embarked"].value_counts().index[i]).astype(int)

    train = train.drop("Embarked", axis=1)
    train = train.drop("Ticket_f", axis=1)

    r = 512.3292
    for i in range(len(train)):
        train.at[i, "Fare"] /= r

    r = 79.58
    for i in range(len(train)):
        train.at[i, "Age"] /= r
        if (not isTest):
            a = train["Survived"].copy()
            train = train.drop("Survived", axis=1)
            
            return train, a
        else:
            return train


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
            nn.Linear(76, 2),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Softmax(dim=0)
        )
        self.double()
    def forward(self, x):
        return (self.stack(x))


x_train, y_train, x_train_id, x_val, y_val, x_val_id, x_test, y_test, x_test_id = splitData()

train = customSet(tensor=[x_train, y_train])
val = customSet(tensor=[x_val, y_val])
test = customSet(tensor=[x_test, y_test])

device = "cpu"

train_dataloader = DataLoader(train, batch_size=3, shuffle=True)
val_dataloader = DataLoader(val, batch_size=3, shuffle=True)
test_dataloader = DataLoader(test, batch_size=3, shuffle=True)

model = network()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.01)

# print(x_train[0])

def trainEpoch():

    running_loss = 0
    last_loss = 0
    prev_param = list(model.parameters())
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        # print(labels)
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

tester = pd.read_csv(os.path.join("test.csv"))

tester = dataManaging(tester, True)



ids = tester["PassengerId"].copy()
tester = tester.drop("PassengerId", axis=1)
tester = torch.from_numpy(tester.to_numpy())
sub = open("file.csv", "w")
sub.write("PassengerId,Survived\n")
ids = ids.to_numpy()
for i in range(len(tester)):
    predict = model(tester[i])
    if tester[i][0] > tester[i][1]:
        sub.write(f"{ids[i]},0\n")
    else:
        sub.write(f"{ids[i]},1\n")