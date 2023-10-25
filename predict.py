import os
import numpy as np
import pandas as pd
import math
import torch

test = pd.read_csv(os.path.join("test.csv"))

test = dataManaging(test, True)

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


predictor = network()