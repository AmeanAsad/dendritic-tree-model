# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:55:13 2022

@author: Amean
"""

import torch
import torch.nn as nn
from pathlib import Path
from dataLoader import getDataset
import matplotlib.pyplot as plt


data = getDataset()
device = torch.device("cpu")

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1278, 1),
          
        )
    def forward(self, x):
        x = x.float()
        res = self.net(x)
#         print(res)
        return res.to(dtype=torch.float64)

if __name__ == "__main__":

    modelPath = Path("mode.pth")
    
    model = torch.load(modelPath) 
    model.eval()
    model.to(device)
    print(model)
    limit = 200;
    counter = 0
    truthVal = []
    pred = []
    while counter < limit:
        dataInput, out  = data[counter]
        truthVal.append(float(out))
        dataInput = torch.transpose(dataInput, 0, 1)
        inputVal = dataInput.float()
        modelres = model(inputVal)
        pred.append(float(modelres))
        counter+=1
        
    plt.figure()
    plt.plot(truthVal, "-r")
    plt.plot(pred)
    plt.show()