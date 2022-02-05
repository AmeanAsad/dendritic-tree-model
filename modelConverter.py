#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:56:58 2022

@author: ameanasad
"""


from pathlib import Path
import json
import numpy as np
from json import JSONEncoder
from tensorflowjs.read_weights import read_weights

import subprocess


class NumpyArrayEncoder(JSONEncoder):
    """
    This class extension modifies the default
    JSON encoder to encode numpy arrays by changing
    them into lists. We need to do this because the
    tensorflow weight files have Numpy arrays nested
    in them.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Path.cwd() gives the current directory of this file.
path = Path().cwd() / "data/Models"
dataPath = Path().cwd() / "data"

# Gets all the file names that end with .h5 which are the model files
modelPaths = [p.name for p in list(path.glob("*.h5"))]


# Loops over each file to conver the file from .h5 to .json format
for fileName in modelPaths:
    modelPath = fileName
    savePath = fileName.replace(".h5", "")

    # subprocess is  a library that allows to run command line scripts
    # in Python code.
    subprocess.Popen(
        [
              "tensorflowjs_converter",
              "--input_format=keras",
              modelPath,
              savePath
        ],
        cwd=path
    )
    print("Converted model {} successfully".format(fileName))


groups = []

print("\n \n")

# Loops over the file names, converts the weights to json objects
# Saves the parsed weights and models in a separate file.
for fileName in modelPaths:
    directory = fileName.removesuffix(".h5")

    jsonModelPath = path / directory / "model.json"
    modelJson = jsonModelPath.open(mode="r")
    model = json.load(modelJson)
    modelManifest = model["weightsManifest"]

    binaryFilePath = path / directory
    weights = read_weights(modelManifest, binaryFilePath, flatten=True)

    saveDirectory = dataPath / "ParsedModels" / directory
    saveDirectory.mkdir(parents=True, exist_ok=True)

    saveModelPath = saveDirectory / "model.json"
    saveWeightsPath = saveDirectory / "weights.json"

    with open(saveModelPath, "w") as modelFile:
        json.dump(model, modelFile, cls=NumpyArrayEncoder)

    with open(saveWeightsPath, "w") as weightsFile:
        json.dump(weights, weightsFile, cls=NumpyArrayEncoder)

    print("Saved Model {} successfully".format(directory))
