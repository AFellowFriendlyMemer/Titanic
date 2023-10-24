import os
import numpy as np
import pandas as pd
import math
from data import dataManaging
import torch
from splitset import splitData

x_train, y_train, x_train_id, x_val, y_val, x_val_id, x_test, y_test, x_test_id = splitData()

device = "cpu"





