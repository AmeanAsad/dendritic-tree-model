# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:01:11 2022

@authors: Amean Asad,
"""


from pathlib import Path
import pickle
import pandas as pd
import numpy as np

dataFile = Path("data/simulations")

modelPaths = [p for p in list(dataFile.glob("*.p"))]

currentPath = modelPaths[0]
data = currentPath.open(mode="rb")
data = pickle.load(data, encoding='latin1')

dataResults = data["Results"]["listOfSingleSimulationDicts"]
dataParams = data["Params"]

dataResult0 = dataResults[0]

numDataPoints = dataParams["totalSimDurationInSec"]*1000
numOfSimulations = len(dataResults)



X = np.zeros((639, numDataPoints, numOfSimulations))
