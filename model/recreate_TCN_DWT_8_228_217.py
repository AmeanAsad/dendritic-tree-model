import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tensorflow as tf
from tensorflow import keras
from pathlib import Path




# Change the directory to fetch the .h5 files
dataFile = Path("../data/Models/NMDA_TCN__DWT_8_224_217__model.h5")

# Load the Keras model and show its architecture

# Read the .h5 model that corresponds to the correct model architecture
model = keras.models.load_model(dataFile)
model.summary()
weights = model.get_weights()


# Define the Pytorch Architecture (Inspired by the Keras Model)


# Define Causal Convolution 1D (no built-in padding style like this exists in Pytorch)
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

# Old Causal CONV1D Definition - May need it in the future if above doesn't work
# def CausalConv1d(in_channels, out_channels, kernel_size, stride=(1,), dilation=(1,), groups=1, bias=True):
#     pad = (kernel_size - 1)*dilation
#     return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = CausalConv1d(in_channels=1278, out_channels=512, kernel_size=3, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv2 = CausalConv1d(in_channels=512, out_channels=384, kernel_size=5, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv3 = CausalConv1d(in_channels=384, out_channels=256, kernel_size=7, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv4 = CausalConv1d(in_channels=256, out_channels=128, kernel_size=9, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv5 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv6 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=39, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv7 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=59, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv8 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=83, stride=(
            1,), dilation=1, groups=1, bias=True)

        # batch normalization layers
        self.batch1 = nn.BatchNorm1d(num_features=500, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch2 = nn.BatchNorm1d(num_features=384, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch3 = nn.BatchNorm1d(num_features=256, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch4 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch5 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch6 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch7 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch8 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)

        # output predictions
        self.spikes = CausalConv1d(in_channels=128, out_channels=1, kernel_size=1, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.soma = CausalConv1d(in_channels=128, out_channels=1, kernel_size=1, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.dendrites = CausalConv1d(in_channels=128, out_channels=64, kernel_size=1, stride=(
            1,), dilation=1, groups=1, bias=True)

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
        x = self.conv8(x)
        x = F.relu(x)
        x = self.batch8(x)

        spikes = self.spikes(x)
        spikes = F.sigmoid(spikes)
        soma = self.soma(x)
        soma = F.linear(soma)
        dendrites = self.dendrites(x)
        dendrites = F.linear(dendrites)

        return spikes, soma, dendrites


net = Net()


# Compare the Keras and Pytorch Weights to verify correctness of defined Pytorch model
print(f'keras weight {weights[0].shape} and Pytorch weight {net.conv1.weight.shape}')
print(f'keras weight {weights[1].shape} and Pytorch weight {net.conv1.bias.shape}')
print(f'keras weight {weights[6].shape} and Pytorch weight {net.conv2.weight.shape}')
print(f'keras weight {weights[8].shape} and Pytorch weight {net.batch2.weight.shape}')
print(f'keras weight {weights[12].shape} and Pytorch weight {net.conv3.weight.shape}')
print(f'keras weight {weights[14].shape} and Pytorch weight {net.batch3.weight.shape}')
print(f'keras weight {weights[18].shape} and Pytorch weight {net.conv4.weight.shape}')
print(f'keras weight {weights[20].shape} and Pytorch weight {net.batch4.weight.shape}')
print(f'keras weight {weights[24].shape} and Pytorch weight {net.conv5.weight.shape}')
print(f'keras weight {weights[26].shape} and Pytorch weight {net.batch5.weight.shape}')
print(f'keras weight {weights[30].shape} and Pytorch weight {net.conv6.weight.shape}')
print(f'keras weight {weights[32].shape} and Pytorch weight {net.batch6.weight.shape}')
print(f'keras weight {weights[36].shape} and Pytorch weight {net.conv7.weight.shape}')
print(f'keras weight {weights[38].shape} and Pytorch weight {net.batch7.weight.shape}')
print(f'keras weight {weights[42].shape} and Pytorch weight {net.conv8.weight.shape}')
print(f'keras weight {weights[44].shape} and Pytorch weight {net.batch8.weight.shape}')
print(f'keras weight {weights[48].shape} and Pytorch weight {net.spikes.weight.shape}')
print(f'keras weight {weights[50].shape} and Pytorch weight {net.soma.weight.shape}')
print(f'keras weight {weights[52].shape} and Pytorch weight {net.dendrites.weight.shape}')


# Load the weights into Pytorch Model
# Transpose the weights, but not the biases (Keras vs Pytorch syntax)
# Convolutional Layer 1
net.conv1.weight.data = torch.from_numpy(np.transpose(weights[0]))
net.conv1.bias.data = torch.from_numpy(weights[1])

# Batch Layer 1
net.batch1.weight.data = torch.from_numpy(weights[2])
net.batch1.bias.data = torch.from_numpy(weights[3])
net.batch1.running_mean.data = torch.from_numpy(weights[4])
net.batch1.running_var.data = torch.from_numpy(weights[5])


# Convolutional Layer 2
net.conv2.weight.data = torch.from_numpy(np.transpose(weights[6]))
net.conv2.bias.data = torch.from_numpy(weights[7])

# Batch Layer 2
net.batch2.weight.data = torch.from_numpy(weights[8])
net.batch2.bias.data = torch.from_numpy(weights[9])
net.batch2.running_mean.data = torch.from_numpy(weights[10])
net.batch2.running_var.data = torch.from_numpy(weights[11])


# Convolutional Layer 3
net.conv3.weight.data = torch.from_numpy(np.transpose(weights[12]))
net.conv3.bias.data = torch.from_numpy(weights[13])

# Batch Layer 3
net.batch3.weight.data = torch.from_numpy(weights[14])
net.batch3.bias.data = torch.from_numpy(weights[15])
net.batch3.running_mean.data = torch.from_numpy(weights[16])
net.batch3.running_var.data = torch.from_numpy(weights[17])

# Convolutional Layer 4
net.conv4.weight.data = torch.from_numpy(np.transpose(weights[18]))
net.conv4.bias.data = torch.from_numpy(weights[19])

# Batch Layer 4
net.batch4.weight.data = torch.from_numpy(weights[20])
net.batch4.bias.data = torch.from_numpy(weights[21])
net.batch4.running_mean.data = torch.from_numpy(weights[22])
net.batch4.running_var.data = torch.from_numpy(weights[23])


# Convolutional Layer 5
net.conv5.weight.data = torch.from_numpy(np.transpose(weights[24]))
net.conv5.bias.data = torch.from_numpy(weights[25])

# Batch Layer 5
net.batch5.weight.data = torch.from_numpy(weights[26])
net.batch5.bias.data = torch.from_numpy(weights[27])
net.batch5.running_mean.data = torch.from_numpy(weights[28])
net.batch5.running_var.data = torch.from_numpy(weights[29])

# Convolutional Layer 6
net.conv6.weight.data = torch.from_numpy(np.transpose(weights[30]))
net.conv6.bias.data = torch.from_numpy(weights[31])

# Batch Layer 6
net.batch6.weight.data = torch.from_numpy(weights[32])
net.batch6.bias.data = torch.from_numpy(weights[33])
net.batch6.running_mean.data = torch.from_numpy(weights[34])
net.batch6.running_var.data = torch.from_numpy(weights[35])

# Convolutional Layer 7
net.conv7.weight.data = torch.from_numpy(np.transpose(weights[36]))
net.conv7.bias.data = torch.from_numpy(weights[37])

# Batch Layer 7
net.batch7.weight.data = torch.from_numpy(weights[38])
net.batch7.bias.data = torch.from_numpy(weights[39])
net.batch7.running_mean.data = torch.from_numpy(weights[40])
net.batch7.running_var.data = torch.from_numpy(weights[41])

# Convolutional Layer 8
net.conv8.weight.data = torch.from_numpy(np.transpose(weights[42]))
net.conv8.bias.data = torch.from_numpy(weights[43])

# Batch Layer 8
net.batch8.weight.data = torch.from_numpy(weights[44])
net.batch8.bias.data = torch.from_numpy(weights[45])
net.batch8.running_mean.data = torch.from_numpy(weights[46])
net.batch8.running_var.data = torch.from_numpy(weights[47])

# Spike Output Layer
net.spikes.weight.data = torch.from_numpy(np.transpose(weights[48]))
net.spikes.bias.data = torch.from_numpy(weights[49])

# Soma Output Layer
net.soma.weight.data = torch.from_numpy(np.transpose(weights[50]))
net.soma.bias.data = torch.from_numpy(weights[51])

# Dendrite Output Layer
net.dendrites.weight.data = torch.from_numpy(np.transpose(weights[52]))
net.dendrites.bias.data = torch.from_numpy(weights[53])
