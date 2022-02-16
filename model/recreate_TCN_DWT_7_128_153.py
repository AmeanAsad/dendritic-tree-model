import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd
import IPython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path


# Change the directory to fetch the .h5 files



#Read the .h5 model that corresponds to the correct model architecture
#model = keras.models.load_model('/NMDA_TCN__DWT_7_128_153__model.h5')
#weights = model.get_weights()

# Define the Pytorch Architecture (Inspired by the Keras Model)
def CausalConv1d(in_channels, out_channels, kernel_size, stride = (1,), dilation = (1,), groups = 1, bias = True):
    pad = (kernel_size -1)*dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = pad, dilation = dilation, groups = groups, bias = bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #convolutional layers
        self.conv1 = CausalConv1d(in_channels = 1278, out_channels = 400, kernel_size = 45, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.conv2 = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 19, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.conv3 = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 19, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.conv4 = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 19, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.conv5 = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 19, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.conv6 = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 19, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.conv7 = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 19, stride = (1,) , dilation = 1, groups = 1, bias = True)
        
        #batch normalization layers
        self.batch1 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        self.batch2 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        self.batch3 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        self.batch4 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        self.batch5 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        self.batch6 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        self.batch7 = nn.BatchNorm1d(num_features = 128, eps = 0.001, momentum = 0.99, affine = True, track_running_stats = True, device = None, dtype = None)
        
        #output predictions
        self.spikes = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 1, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.soma = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 1, stride = (1,) , dilation = 1, groups = 1, bias = True)
        self.dendrites = CausalConv1d(in_channels = 128, out_channels = 400, kernel_size = 1, stride = (1,) , dilation = 1, groups = 1, bias = True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batch4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batch5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batch6(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.batch7(x)
        
        spikes = self.spikes(x)
        spikes = F.sigmoid(spikes)
        soma = self.soma(x)
        soma = F.linear(soma)
        dendrites = self.dendrites(x)
        dendrites = F.linear(dendrites)
        
        return spikes, soma, dendrites

# Compare the Keras and Pytorch Weights to verify correctness of defined Pytorch model

#Load the weights into Pytorch Model
#Transpose the weights, but not the biases (Keras vs Pytorch syntax)
net.conv1.weight.data=torch.from_numpy(np.transpose(weights[0]))
net.batch1.weight.data=torch.from_numpy(weights[1])
net.conv2.weight.data=torch.from_numpy(np.transpose(weights[2]))
net.batch2.weight.data=torch.from_numpy(weights[3])
net.conv3.weight.data=torch.from_numpy(np.transpose(weights[4]))
net.batch3.weight.data=torch.from_numpy(weights[5])
net.conv4.weight.data=torch.from_numpy(np.transpose(weights[6]))
net.batch4.weight.data=torch.from_numpy(weights[7])
net.conv5.weight.data=torch.from_numpy(np.transpose(weights[8]))
net.batch5.weight.data=torch.from_numpy(weights[9]])
net.conv6.weight.data=torch.from_numpy(np.transpose(weights[10]))
net.batch6.weight.data=torch.from_numpy(weights[11])
net.conv7.weight.data=torch.from_numpy(np.transpose(weights[12]))
net.batch7.weight.data=torch.from_numpy(weights[13])

