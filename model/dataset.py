# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:34:46 2022

@author: ericv, Amean Asad

Data comes in as an array:
    639 inhibitory + 639 excitatory neurons as columns + 1 column for Spike = 0/1
    6000ms * 128experiements as rows
    (768000 x 1279)
    filled with 1s and 0s (active or not)
"""

import torch
from torch.utils.data import Dataset


# Custom Dataset
class SimulationDatasetFCN(Dataset):
    def __init__(self, inputData, outputData, windowSize):
        self.inputData = torch.tensor(inputData)
        self.windowSize = windowSize
        self.outputData = torch.tensor(outputData)

    def __len__(self):
        shape = self.inputData.shape    
        return shape[2] - (self.windowSize + 1)

    # Add whatever will need to be retrieved from the data
    def __getitem__(self, idx):
        # Assuming spike times are listed in the first column of the data, 
        # grabs the spike value (0/1) at a specific time
        endIdx = self.windowSize + idx
        # Change this when switching between 1d and 2d data.
        inputItem = self.inputData[0,:, idx:endIdx]
        outputItem = self.outputData[endIdx, 0]

        return inputItem, outputItem


class SimulationDatasetTCN(Dataset):
    def __init__(self, inputData, outputData, windowSize):
        self.inputData = torch.tensor(inputData)
        self.windowSize = windowSize
        self.outputData = torch.tensor(outputData)
        shape = self.inputData.shape    
        self.length =  int(shape[2]/self.windowSize)
        print("length", shape[2])
        print(self.length)
        

    def __len__(self):
      
        return self.length

    # Add whatever will need to be retrieved from the data
    def __getitem__(self, idx):
        # Assuming spike times are listed in the first column of the data, 
        # grabs the spike value (0/1) at a specific time
        
        if idx > self.length:
            raise IndexError("Index too high")
        
        startIdx = self.windowSize*(idx) - 150*idx
        endIdx = startIdx + self.windowSize
        outputStartIdx = endIdx - 250
    
        # Change this when switching between 1d and 2d data.
        inputItem = self.inputData[0,:, startIdx:endIdx]
        outputItem = self.outputData[outputStartIdx:endIdx, 0]

        return inputItem, outputItem