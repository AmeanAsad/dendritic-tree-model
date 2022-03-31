# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:56:20 2022

@author: Amean
"""

from torch.utils.data import DataLoader
from dataLoader import getDataset
# from recreate_TCN_DWT_7_292_169 import Net
import torch

data = getDataset()



if __name__ == "__main__":
    testData = DataLoader(data, batch_size=64, num_workers = 1)
    # print(testData[0])
    # for batch in testData:
    #     print(batch)
#     network = Net()
#     network = network.float()
#     with torch.no_grad():
#         count = 0
#         correct = 0
#         total = 0
#         for i, data in enumerate(testData, 0):           
#             count +=1
#             # Get inputs
#             inputs, targets = data
#             print("targets", targets.shape)
#             print("inputs ", inputs.shape)
#             # Generate outputs
#             outputs = network(inputs.float())
#             print("output", outputs.shape)
#             # Set total and correct
#             _, predicted = torch.max(outputs.data, 1)
#             print("Predictions", predicted.shape)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#             if count > 50:
#                 break
#         # Print accuracy
#         print('Accuracy: %d %%' % (100 * correct / total))
