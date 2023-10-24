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

for i in range(train["Cabin_Letter"].value_counts()):

