import os
import numpy as np
import pandas as pd
import math

import sklearn

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

    for i in range(1, len(train["Cabin_Letter"].value_counts())):
        train["Cabin_" + train["Cabin_Letter"].value_counts().index[i]] = (train["Cabin_Letter"] == train["Cabin_Letter"].value_counts().index[i]).astype(int)

    train = train.drop("Cabin_Letter", axis=1)
    train = train.drop("Cabin", axis=1)
    age_median = train["Age"].describe()["50%"]
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

    for i in range(1, len(train["Ticket_f"].value_counts().index)):
        train["Ticket_f_" + train["Ticket_f"].value_counts().index[i]] = (train["Ticket_f"] == train["Ticket_f"].value_counts().index[i]).astype(int)

    train = train.drop("Ticket", axis=1)

    for i in range(len(train)):
        n = train.at[i, "Name"]
        n = n[n.find(','):n.find('.')][2:]
        train.at[i, "Name"] = n

    for i in range(1, len(train["Name"].value_counts().index)):
        train["Name_" + train["Name"].value_counts().index[i]] = (train["Name"] == train["Name"].value_counts().index[i]).astype(int)

    train = train.drop("Name", axis=1)

    for i in range(0, len(train["Embarked"].value_counts().index)):
        train["Embarked_" + train["Embarked"].value_counts().index[i]] = (train["Embarked"] == train["Embarked"].value_counts().index[i]).astype(int)

    train = train.drop("Embarked", axis=1)
    train = train.drop("Ticket_f", axis=1)

    r = train["Fare"].describe()["max"] - train["Fare"].describe()["min"]

    for i in range(len(train)):
        train.at[i, "Fare"] /= r

    r = train["Age"].describe()["max"] - train["Age"].describe()["min"]

    for i in range(len(train)):
        train.at[i, "Age"] /= r

    if (not isTest):
        a = train["Survived"].copy()
        train = train.drop("Survived", axis=1)
        return train, a
    else:
        return train


    
    