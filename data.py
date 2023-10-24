import os
import numpy as np
import pandas as pd
import math

train = pd.read_csv(os.path.join("train.csv"))

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

for i in range(1, len(train["Cabin_Letter"].value_counts())):
    train["Cabin_" + train["Cabin_Letter"].value_counts().index[i]] = (train["Cabin_Letter"] == train["Cabin_Letter"].value_counts().index[i]).astype(int)

train.drop("Cabin_Letter", axis=1)
train.drop("Cabin", axis=1)
age_median = train["Age"].describe()["50%"]
for i in range(len(train)):
    if math.isnan(train.iloc[i]["Age"]):
        train.at[i, "Age"] = age_median

train["Sex"] = (train["Sex"] == "male").astype(int)

ticket_f = []

for i in range(len(train)):
    ticket_f.append(train.at[i, "Ticket"].split(" ")[0]) #fix

print(ticket_f)
