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


def spikeDictToArray(spikeTimes, numOfSegments, numDataPoints):
    
    spikeValuesMatrix = np.zeros((numOfSegments, numDataPoints))
    spikeTimeVals = spikeTimes.values()
    
    for idx, spikeArray in enumerate(spikeTimeVals):
        for spikeTime in spikeArray:
            spikeValuesMatrix[idx, spikeTime] = 1.0
    return spikeValuesMatrix
    
def parseSimulationFile(filePath):
    """
    Parameters
    ----------
    filePath : Pathlib Path
        A filepath to a pickled simulation file generated.
        The filepath must be a Pathlib path for the function to work. 
    Returns
    -------
    Parsed 

    """
    data = filePath.open(mode="rb")
    data = pickle.load(data, encoding='latin1')

    dataResults = data["Results"]["listOfSingleSimulationDicts"][:1]
    dataParams = data["Params"]

    numDataPoints = dataParams["totalSimDurationInSec"]*1000
    numOfSimulations = len(dataResults)
    numOfSegments = len(dataParams["allSegmentsType"])
    synapseCount = 639 * 2 # 639 Inhibitory + 639 Excitatory inputs
    

    X = np.zeros((synapseCount, numDataPoints, numOfSimulations))
    spikeVals = np.zeros((numDataPoints, numOfSimulations))
    somaVoltages = np.zeros((numDataPoints, numOfSimulations))
    nexusVoltages = np.zeros((numDataPoints, numOfSimulations))
    
    dendriticVoltages  = np.zeros(
        (numOfSegments, numDataPoints,numOfSimulations), dtype=np.float16)

    
    for idx, simulationResult in enumerate(dataResults):
    
        inhibitorySpikes = spikeDictToArray(simulationResult["inhInputSpikeTimes"],
                                    numOfSegments, numDataPoints)
        excitatorySpikes = spikeDictToArray(simulationResult["exInputSpikeTimes"],
                                    numOfSegments, numDataPoints)
        
        X[:,:, idx] = np.vstack((excitatorySpikes, inhibitorySpikes))
        dendriticVoltages[:,:,idx] = simulationResult["dendriticVoltagesLowRes"]
            
        spikeTimes = (simulationResult['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        spikeVals[spikeTimes, idx] = 1.0
        
        somaVoltages[:,idx] = simulationResult["somaVoltageLowRes"]
        nexusVoltages[:,idx] = simulationResult["nexusVoltageLowRes"]
        
        somaVoltages[spikeTimes, idx] = 30

    
    return X, spikeVals, somaVoltages, nexusVoltages, dendriticVoltages, dataResults
    
    
X, spike, soma, n, v, d = parseSimulationFile(modelPaths[0])

