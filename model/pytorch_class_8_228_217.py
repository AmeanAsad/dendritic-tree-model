import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1 = CausalConv1d(in_channels=1278, out_channels=512, kernel_size=(3,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv2 = CausalConv1d(in_channels=512, out_channels=384, kernel_size=(5,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv3 = CausalConv1d(in_channels=384, out_channels=256, kernel_size=(7,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv4 = CausalConv1d(in_channels=256, out_channels=128, kernel_size=(9,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv5 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=(19,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv6 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=(39,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv7 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=(59,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.conv8 = CausalConv1d(in_channels=128, out_channels=128, kernel_size=(83,), stride=(
            1,), dilation=(1,), groups=1, bias=True)

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
        self.spikes = CausalConv1d(in_channels=128, out_channels=1, kernel_size=(1,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.soma = CausalConv1d(in_channels=128, out_channels=1, kernel_size=(1,), stride=(
            1,), dilation=(1,), groups=1, bias=True)
        self.dendrites = CausalConv1d(in_channels=128, out_channels=64, kernel_size=(1,), stride=(
            1,), dilation=(1,), groups=1, bias=True)

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