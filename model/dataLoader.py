# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:01:11 2022

@authors: Amean Asad, Eric Venditti
"""


from pathlib import Path
import pickle
import numpy as np
from timeit import default_timer as timer
from dataset import SimulationDatasetFCN,SimulationDatasetTCN
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

dataFile = Path("../data/simulations")

modelPaths = [p for p in list(dataFile.glob("*.p"))]

# print("------------------", modelPaths)
print("Files to render: {num}".format(num=len(modelPaths)))

def spikeDictToArray(spikeTimes, numOfSegments, numDataPoints):

    spikeValuesMatrix = np.zeros((numOfSegments, numDataPoints))
    spikeTimeVals = spikeTimes.values()

    for idx, spikeArray in enumerate(spikeTimeVals):
        for spikeTime in spikeArray:
            spikeValuesMatrix[idx, spikeTime] = 1.0
    return spikeValuesMatrix


def aggregateMultipleFiles(modelPaths):
    
    X, soma, nexus, dvt, spike = parseSimulationFileForModel(modelPaths[0], numOfSims=90)
    
    for file in modelPaths[1:]:
        
        XTemp, somaTemp, nexusT, dvtT, spikeT = parseSimulationFileForModel(file, numOfSims=90)
        
        X = np.concatenate((X, XTemp), axis=2)
        soma = np.concatenate((soma, somaTemp), axis=0)
    return X, soma


def parseSimulationFile(filePath):
    """
    Parameters
    ----------
    filePath : Pathlib Path
        A filepath to a pickled simulation file generated.
        The filepath must be a Pathlib path for the function to work.
    Returns
    -------
    X: Array representing the inputs at each time stamp
    spikeVals: Array of boolean spike values at each time stamp
    somaVoltages: Array of soma voltages at each time stamp
    nexusVoltages: Array of nexus voltages at each time stamp
    dendriticVoltages: Array of dendritic voltages at each time stamp
    """

    print("Currently parsing file: {}".format(filePath.name))
    startTime = timer()

    data = filePath.open(mode="rb")
    data = pickle.load(data, encoding='latin1')

    dataResults = data["Results"]["listOfSingleSimulationDicts"][:2]
    dataParams = data["Params"]

    numDataPoints = dataParams["totalSimDurationInSec"]*1000
    numOfSimulations = len(dataResults)

    #  allSegmentsType is a list (length 639) half "basal"/half "apical"
    numOfSegments = len(dataParams["allSegmentsType"])
    synapseCount = 639 * 2  # 639 Inhibitory + 639 Excitatory inputs

    # 1278 x 6000 x 128 multidimensional array
    X = np.zeros((synapseCount, numDataPoints, numOfSimulations))
    spikeVals = np.zeros((numDataPoints, numOfSimulations))
    somaVoltages = np.zeros((numDataPoints, numOfSimulations))
    nexusVoltages = np.zeros((numDataPoints, numOfSimulations))

    dendriticVoltages = np.zeros(
        (numOfSegments, numDataPoints, numOfSimulations), dtype=np.float16)

    for idx, simulationResult in enumerate(dataResults):

        inhibitorySpikes = spikeDictToArray(simulationResult["inhInputSpikeTimes"],
                                            numOfSegments, numDataPoints)
        excitatorySpikes = spikeDictToArray(simulationResult["exInputSpikeTimes"],
                                            numOfSegments, numDataPoints)

        X[:, :, idx] = np.vstack((excitatorySpikes, inhibitorySpikes))
        # dendriticVoltages[:, :, idx] = simulationResult["dendriticVoltagesLowRes"]

        spikeTimes = (simulationResult['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        spikeVals[spikeTimes, idx] = 1.0

        somaVoltages[:, idx] = simulationResult["somaVoltageLowRes"]
        nexusVoltages[:, idx] = simulationResult["nexusVoltageLowRes"]

        somaVoltages[spikeTimes, idx] = 30

    # 6000x1 counting array, increments of 1
    timeStamps = simulationResult["recordingTimeLowRes"]

    endTime = timer() - startTime
    print("Done, time elapsed: {} \n".format(round(endTime, 3)))

    return X, spikeVals, somaVoltages, nexusVoltages, dendriticVoltages, timeStamps


def parseSimulationFileForModel(filePath, numOfSims=10):
    """
    Parameters
    ----------
    filePath : Pathlib Path
        A filepath to a pickled simulation file generated.
        The filepath must be a Pathlib path for the function to work.
    Returns
    -------
    X: Array representing the inputs at each time stamp
    spikeVals: Array of boolean spike values at each time stamp
    somaVoltages: Array of soma voltages at each time stamp
    nexusVoltages: Array of nexus voltages at each time stamp
    dendriticVoltages: Array of dendritic voltages at each time stamp
    """

    print("Currently parsing file: {}".format(filePath.name))
    startTime = timer()

    data = filePath.open(mode="rb")
    data = pickle.load(data, encoding='latin1')

    dataResults = data["Results"]["listOfSingleSimulationDicts"][:numOfSims]
    dataParams = data["Params"]

    numDataPoints = dataParams["totalSimDurationInSec"]*1000
    numOfSimulations = len(dataResults)
    numOfSegments = len(dataParams["allSegmentsType"])
    synapseCount = 639 * 2  # 639 Inhibitory + 639 Excitatory inputs

    X = np.zeros((1, synapseCount, numDataPoints*numOfSimulations), dtype=np.float16)
    print(X.shape)
    # spikeVals = np.zeros((numDataPoints*numOfSimulations, 1), dtype=np.float16)
    somaVoltages = np.zeros((numDataPoints*numOfSimulations, 1))
    # nexusVoltages = np.zeros((numDataPoints*numOfSimulations))

    dendriticVoltages = np.zeros(
        (numOfSegments, numDataPoints*numOfSimulations), dtype=np.float16)

    for idx, simulationResult in enumerate(dataResults):

        inhibitorySpikes = spikeDictToArray(simulationResult["inhInputSpikeTimes"],
                                            numOfSegments, numDataPoints)
        excitatorySpikes = spikeDictToArray(simulationResult["exInputSpikeTimes"],
                                            numOfSegments, numDataPoints)
        startIdx = numDataPoints*idx
        endIdx = numDataPoints*(idx + 1)
        X[0, :, startIdx: endIdx] = np.vstack((excitatorySpikes, inhibitorySpikes))
        # dendriticVoltages[:, startIdx:endIdx] = simulationResult["dendriticVoltagesLowRes"]

        spikeTimes = (simulationResult['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        # spikeVals[spikeTimes + startIdx, 0] = 1.0

        somaVoltages[startIdx:endIdx,0] = simulationResult["somaVoltageLowRes"]
        # nexusVoltages[startIdx:endIdx] = simulationResult["nexusVoltageLowRes"]

        # somaVoltages[spikeTimes + startIdx] = 30
    
    somaVoltages[somaVoltages[:,0] > -55] = -55
    somaVoltages[:,0] = somaVoltages[:,0] + 67.7


    # timeStamps = simulationResult["recordingTimeLowRes"]

    endTime = timer() - startTime
    print("Done, time elapsed: {} \n".format(round(endTime, 3)))

    return X, somaVoltages, [], [],[]


def getDatasetForTCN(numOfSims=10):
    X, soma, nexus, dvt, spike = parseSimulationFileForModel(modelPaths[0], numOfSims)
    
    dataLength = soma.shape[0]
    testLength = int(dataLength*0.85)
    testX = X[:,:, testLength:] 
    testSoma = soma[testLength:,:]
    
    trainX = X[:,:, :testLength]
    trainSoma = soma[:testLength,:]

  
    return SimulationDatasetTCN(trainX, trainSoma, windowSize=400), SimulationDatasetTCN(testX, testSoma, windowSize=400)

def getDatasetForTCN2():
    X, soma = aggregateMultipleFiles(modelPaths)
    
    dataLength = soma.shape[0]
    testLength = int(dataLength*0.85)
    testX = X[:,:, testLength:] 
    testSoma = soma[testLength:,:]
    
    trainX = X[:,:, :testLength]
    trainSoma = soma[:testLength,:]

  
    return SimulationDatasetTCN(trainX, trainSoma, windowSize=400), SimulationDatasetTCN(testX, testSoma, windowSize=400)


def getDatasetForFCN(numOfSims=10):
    X, soma, nexus, dvt, spike = parseSimulationFileForModel(modelPaths[0], numOfSims)
    
    dataLength = soma.shape[0]
    testLength = int(dataLength*0.85)
    testX = X[:,:, testLength:] 
    testSoma = soma[testLength:,:]
    
    trainX = X[:,:, :testLength]
    trainSoma = soma[:testLength,:]
   
    return SimulationDatasetFCN(trainX, trainSoma, windowSize=1), SimulationDatasetFCN(testX, testSoma, windowSize=1)

    
# X, soma = aggregateMultipleFiles(modelPaths)
# # print(X.shape)
# print(soma.shape)
