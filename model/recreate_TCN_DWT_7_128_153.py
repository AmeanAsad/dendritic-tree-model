import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tensorflow as tf
from tensorflow import keras
from pathlib import Path


# Change the directory to fetch the .h5 files
dataFile = Path("../data/Models/NMDA_TCN__DWT_7_128_153__model.h5")

# Read the .h5 model that corresponds to the correct model architecture
model = keras.models.load_model(dataFile)
weights = model.get_weights()

# Define the Pytorch Architecture (Inspired by the Keras Model)
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

# def CausalConv1d(in_channels, out_channels, kernel_size, stride=(1,), dilation=(1,), groups=1, bias=True):
#     pad = (kernel_size - 1)*dilation
#     return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = CausalConv1d(in_channels=1278, out_channels=128, kernel_size=45, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv2 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv3 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv4 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv5 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv6 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.conv7 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=19, stride=(
            1,), dilation=1, groups=1, bias=True)

        # batch normalization layers
        self.batch1 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch2 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch3 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch4 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch5 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch6 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch7 = nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99,
                                     affine=True, track_running_stats=True, device=None, dtype=None)

        # output predictions
        self.spikes = CausalConv1d(in_channels=128, out_channels=128, kernel_size=1, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.soma = CausalConv1d(in_channels=128, out_channels=128, kernel_size=1, stride=(
            1,), dilation=1, groups=1, bias=True)
        self.dendrites = CausalConv1d(in_channels=128, out_channels=128, kernel_size=1, stride=(
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

        spikes = self.spikes(x)
        spikes = F.sigmoid(spikes)
        soma = self.soma(x)
        soma = F.linear(soma)
        dendrites = self.dendrites(x)
        dendrites = F.linear(dendrites)

        return spikes, soma, dendrites

net = Net()
print(net.conv1)
# Compare the Keras and Pytorch Weights to verify correctness of defined Pytorch model
print(f'keras weight {weights[0].shape} and Pytorch weight {net.conv1.weight.shape}')
print(f'keras weight {weights[1].shape} and Pytorch weight {net.conv1.bias.shape}')
print(f'keras weight {weights[6].shape} and Pytorch weight {net.conv2.weight.shape}')
print(f'keras weight {weights[3].shape} and Pytorch weight {net.batch2.weight.shape}')
print(f'keras weight {weights[4].shape} and Pytorch weight {net.conv3.weight.shape}')
print(f'keras weight {weights[5].shape} and Pytorch weight {net.batch3.weight.shape}')
print(f'keras weight {weights[6].shape} and Pytorch weight {net.conv4.weight.shape}')
print(f'keras weight {weights[7].shape} and Pytorch weight {net.batch4.weight.shape}')
print(f'keras weight {weights[8].shape} and Pytorch weight {net.conv5.weight.shape}')
print(f'keras weight {weights[9].shape} and Pytorch weight {net.batch5.weight.shape}')
print(f'keras weight {weights[10].shape} and Pytorch weight {net.conv6.weight.shape}')
print(f'keras weight {weights[11].shape} and Pytorch weight {net.batch6.weight.shape}')
print(f'keras weight {weights[12].shape} and Pytorch weight {net.conv7.weight.shape}')
print(f'keras weight {weights[13].shape} and Pytorch weight {net.batch7.weight.shape}')


# Load the weights into Pytorch Model
# Transpose the weights, but not the biases (Keras vs Pytorch syntax)
net.conv1.weight.data = torch.from_numpy(np.transpose(weights[0]))
net.conv1[0].bias.data = torch.from_numpy(weights[1])
net.conv1[1].bias.data = torch.from_numpy(weights[2])
net.conv1[2].bias.data = torch.from_numpy(weights[3])
net.conv1[3].bias.data = torch.from_numpy(weights[4])
net.conv1[4].bias.data = torch.from_numpy(weights[5])


net.conv2.weight.data = torch.from_numpy(np.transpose(weights[6]))
# net.conv2[0].bias.data = torch.from_numpy(weights[7])
# net.conv2[1].bias.data = torch.from_numpy(weights[8])
# net.conv1[2].bias.data = torch.from_numpy(weights[9])
# net.conv1[3].bias.data = torch.from_numpy(weights[10])
# net.conv1[4].bias.data = torch.from_numpy(weights[11])

net.conv3.weight.data = torch.from_numpy(np.transpose(weights[12]))
# net.conv1[0].bias.data = torch.from_numpy(weights[13])
# net.conv1[1].bias.data = torch.from_numpy(weights[14])
# net.conv1[2].bias.data = torch.from_numpy(weights[15])
# net.conv1[3].bias.data = torch.from_numpy(weights[16])
# net.conv1[4].bias.data = torch.from_numpy(weights[17])

net.conv4.weight.data = torch.from_numpy(np.transpose(weights[18]))
# net.conv1[0].bias.data = torch.from_numpy(weights[19])
# net.conv1[1].bias.data = torch.from_numpy(weights[20])
# net.conv1[2].bias.data = torch.from_numpy(weights[21])
# net.conv1[3].bias.data = torch.from_numpy(weights[22])
# net.conv1[4].bias.data = torch.from_numpy(weights[23])

net.conv5.weight.data = torch.from_numpy(np.transpose(weights[24]))
# net.conv1[0].bias.data = torch.from_numpy(weights[25])
# net.conv1[1].bias.data = torch.from_numpy(weights[26])
# net.conv1[2].bias.data = torch.from_numpy(weights[27])
# net.conv1[3].bias.data = torch.from_numpy(weights[28])
# net.conv1[4].bias.data = torch.from_numpy(weights[29])

net.conv6.weight.data = torch.from_numpy(np.transpose(weights[30]))
# net.conv1[0].bias.data = torch.from_numpy(weights[31])
# net.conv1[1].bias.data = torch.from_numpy(weights[32])
# net.conv1[2].bias.data = torch.from_numpy(weights[33])
# net.conv1[3].bias.data = torch.from_numpy(weights[34])
# net.conv1[4].bias.data = torch.from_numpy(weights[35])

net.conv7.weight.data = torch.from_numpy(np.transpose(weights[36]))
# net.conv1[0].bias.data = torch.from_numpy(weights[37])
# net.conv1[1].bias.data = torch.from_numpy(weights[38])
# net.conv1[2].bias.data = torch.from_numpy(weights[39])
# net.conv1[3].bias.data = torch.from_numpy(weights[40])
# net.conv1[4].bias.data = torch.from_numpy(weights[41])

net.spikes.weight.data = torch.from_numpy(np.transpose(weights[42]))
# net.conv1[0].bias.data = torch.from_numpy(weights[43])


net.soma.weight.data = torch.from_numpy(np.transpose(weights[44]))
# net.conv1[0].bias.data = torch.from_numpy(weights[45])


net.spikes.weight.data = torch.from_numpy(np.transpose(weights[46]))
# net.conv1[0].bias.data = torch.from_numpy(weights[47])


