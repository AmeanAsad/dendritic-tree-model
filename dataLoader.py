# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:01:11 2022

@authors: Amean Asad,
"""


from pathlib import Path
import pickle
import numpy as np
from timeit import default_timer as timer
from dataset import SimulationDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

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
        dendriticVoltages[:, :, idx] = simulationResult["dendriticVoltagesLowRes"]

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


def parseSimulationFileForModel(filePath):
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

    print("Currently parsing file: {}".format(filePath.name))
    startTime = timer()

    data = filePath.open(mode="rb")
    data = pickle.load(data, encoding='latin1')

    dataResults = data["Results"]["listOfSingleSimulationDicts"][4:5]
    dataParams = data["Params"]

    numDataPoints = dataParams["totalSimDurationInSec"]*1000
    numOfSimulations = len(dataResults)
    numOfSegments = len(dataParams["allSegmentsType"])
    synapseCount = 639 * 2  # 639 Inhibitory + 639 Excitatory inputs

    X = np.zeros((synapseCount, numDataPoints*numOfSimulations))
    spikeVals = np.zeros((numDataPoints*numOfSimulations, 1))
    somaVoltages = np.zeros((numDataPoints*numOfSimulations))
    nexusVoltages = np.zeros((numDataPoints*numOfSimulations))

    dendriticVoltages = np.zeros(
        (numOfSegments, numDataPoints*numOfSimulations), dtype=np.float16)

    for idx, simulationResult in enumerate(dataResults):

        inhibitorySpikes = spikeDictToArray(simulationResult["inhInputSpikeTimes"],
                                            numOfSegments, numDataPoints)
        excitatorySpikes = spikeDictToArray(simulationResult["exInputSpikeTimes"],
                                            numOfSegments, numDataPoints)
        startIdx = numDataPoints*idx
        endIdx = numDataPoints*(idx + 1)
        X[:, startIdx: endIdx] = np.vstack((excitatorySpikes, inhibitorySpikes))
        dendriticVoltages[:, startIdx:endIdx] = simulationResult["dendriticVoltagesLowRes"]

        spikeTimes = (simulationResult['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        spikeVals[spikeTimes + startIdx, 0] = 1.0

        somaVoltages[startIdx:endIdx] = simulationResult["somaVoltageLowRes"]
        nexusVoltages[startIdx:endIdx] = simulationResult["nexusVoltageLowRes"]

        somaVoltages[spikeTimes + startIdx] = 30

    # timeStamps = simulationResult["recordingTimeLowRes"]

    endTime = timer() - startTime
    print("Done, time elapsed: {} \n".format(round(endTime, 3)))

    return X, somaVoltages, nexusVoltages, dendriticVoltages, spikeVals


X, soma, nexus, dvt, spike = parseSimulationFileForModel(modelPaths[0])


dataset = SimulationDataset(X, spike, windowSize=150)
print(dataset[0])

t, img = dataset[3000]
print(t.shape)
print(img)
plt.figure()
plt.imshow(t)
data_load = DataLoader(dataset, batch_size=64)
