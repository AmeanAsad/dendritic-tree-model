# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:55:23 2022

@author: ericv
"""

from pathlib import Path
import pickle
import numpy as np

sim_experiment_file = Path(".\sim__saved_InputSpikes_DVTs__561_outSpikes__128_simulationRuns__6_secDuration__randomSeed_100520.p")


# parse_sim_experiment_file() is accounted for in the dataLoader.py file
# This file will have other data loading functions
# Will eventually be merged with dataLoader.py

# holds the dictionary variable (params/results) from the pickle file in Data_test
experiment_dict = pickle.load(open(sim_experiment_file, "rb" ), encoding='latin1')

'''
1.
parse_sim_experiment_file_with_DVTs(sim_experiment_file, return_high_res=False)
DVT = "Dendritic Voltage Traces"
'''
def parse_sim_experiment_file_with_DVTs(sim_experiment_file, return_high_res=False):
    # Using two "_"s as throwaway variables to intentionally ignore the 2nd and 3rd outputs from parse_sim_experiment_file
    # Those being "y_spike" and "y_soma"
    X_spikes, _, _ = parse_sim_experiment_file(sim_experiment_file)
    
    # "gather params"
    
    # "listOfSingleSimulationDicts" is the only key in the "Results" dictionary
    # Its value is a *list* of dictionaries
    # Each of those dictionaries holds the results of a simulation
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    #  allSegmentsType is a list (length 639) half "basal"/half "apical"
    # This is used later to set the number of exitatory and inhibiroty synapses
    num_segments    = len(experiment_dict['Params']['allSegmentsType'])
    # Turn the simulation time into milliseconds
    sim_duration_ms = 1000 * experiment_dict['Params']['totalSimDurationInSec']
    
    # "collect X, y_spike, y_soma"
    # Saves the the first result dictionary in sim_dict
    sim_dict = experiment_dict['Results']['listOfSingleSimulationDicts'][0]
    
    # 6000x1 counting array, increments of 1
    t_LR = sim_dict['recordingTimeLowRes']
    # 6000x1 couting array, increments of 0.125
    t_HR = sim_dict['recordingTimeHighRes']
    # Zero array of size (time of simulation in milliseconds x number of simulations)
    # Should be 6000 x 128
    y_soma_LR  = np.zeros((sim_duration_ms,num_simulations))
    y_nexus_LR = np.zeros((sim_duration_ms,num_simulations))
    # Zero array of size (number of high res soma voltage samples x number of simulations)
    # Both should be 48000 x 128
    y_soma_HR  = np.zeros((sim_dict['somaVoltageHighRes'].shape[0],num_simulations))
    y_nexus_HR = np.zeros((sim_dict['nexusVoltageHighRes'].shape[0],num_simulations))
    
    # 639 x 6000 x 128 multidimensional array
    y_DVTs  = np.zeros((num_segments,sim_duration_ms,num_simulations), dtype=np.float16)
    
    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        y_nexus_LR[:,k] = sim_dict['nexusVoltageLowRes']
        y_soma_LR[:,k] = sim_dict['somaVoltageLowRes']    
        y_nexus_HR[:,k] = sim_dict['nexusVoltageHighRes']
        y_soma_HR[:,k] = sim_dict['somaVoltageHighRes']    
        y_DVTs[:,:,k] = sim_dict['dendriticVoltagesLowRes']
    
        output_spike_times = np.int32(sim_dict['outputSpikeTimes'])
        # fix "voltage spikes" in low res
        y_soma_LR[output_spike_times,k] = 30
        
        # return_high_res in function definition
        return    
        if return_high_res:
            return X_spikes, y_DVTs, t_LR, y_soma_LR, y_nexus_LR, t_HR, y_soma_HR, y_nexus_HR
        else:
            return X_spikes, y_DVTs, t_LR, y_soma_LR, y_nexus_LR