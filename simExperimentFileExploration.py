# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:04:21 2022

@author: ericv
"""

from pathlib import Path
import pickle

# Path to simulation pickle
sim_experiment_file = Path("data\simulations\sim__saved_InputSpikes_DVTs__561_outSpikes__128_simulationRuns__6_secDuration__randomSeed_100520.p")

# pickle contains a dictionary
experiment_dict = pickle.load(open(sim_experiment_file, "rb" ), encoding='latin1')

# The parent dictionary has two keys, 'Params' and 'Results'
# These keys are dictionaries themselves (nested dictionary)
print("The parent keys are: ", experiment_dict.keys())

print("---------------------")

# These are the keys of 'Params' (parameters) for the experiment
experiment_params = experiment_dict["Params"]
print("The keys of child dict, Params:")
for key in experiment_params:
    print(key)

print("---------------------")

# There is only one key in 'Results'
experiment_results = experiment_dict["Results"]
print("The keys of child dict, Results:")
for key in experiment_results:
    print(key)

print("---------------------")

# This one key, 'listOfSingleSimulationDicts', is a list
# A list of dictionaries
experiment_results_key = experiment_results['listOfSingleSimulationDicts']

# The keys of one value in the list of results
# Each of these keys has an array value
print("Keys of a value in the list of result dictionaries: ")
print(experiment_results_key[0].keys())