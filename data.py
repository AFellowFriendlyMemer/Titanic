import os
import numpy as np
import pandas as pd
import math

import sklearn


cabin_num = []

train = pd.read_csv(os.path.join("train.csv"))


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



    
    