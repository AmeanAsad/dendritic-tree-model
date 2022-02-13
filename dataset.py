# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:34:46 2022

@author: ericv

Data comes in as an array:
    639 inhibitory + 639 excitatory neurons as columns + 1 column for Spike = 0/1
    6000ms * 128experiements as rows
    (768000 x 1279)
    filled with 1s and 0s (active or not)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# pathway to input array
inputArrayPath = r''

# Setup input array as a dataframe
inputDF = pd.read_csv(inputArrayPath)
# Labels are all possible spike times
labelsList = list(range(0, 6000 * 128))

# Custom Dataset
class SimulationDataset(Dataset):
    def __init__(self, inputData, rootDir):
        self.inputData = pd.DataFrame(inputData)
        self.rootDir = rootDir

    def __len__(self):
        return len(self.inputData)

    # Add whatever will need to be retrieved from the data
    def __getitem__(self, idx):
        # Assuming spike times are listed in the first column of the data, grabs the spike value (0/1) at a specific time
        spikeValue = os.path.join(self.rootDir, self.inputData.iloc[idx, 0])
        return spikeValue


# Load the training data
training_data = datasets.SimulationData(
    root="data",  # path to train/test data
    train=True,
)

# Load the test data
test_data = SimulationDataset(
    root="data",
    train=False,
)