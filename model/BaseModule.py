# -*- coding: utf-8 -*-
"""
Base Module

@author: Amean Asad
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from dataLoader import getDataset
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

class BaseModule(nn.Module):
    def training_step(self, batch):
        inputs, targets = batch
        out = self(targets)
        loss = F.cross_entropy(out, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        out = self(targets)
        loss = F.cross_entropy(out, targets)
        acc = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def train_val_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = F.cross_entropy(out, targets)
        acc = accuracy(out, targets)
        return {'train_loss': loss.detach(), 'train_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(epoch + 1, result['val_loss'], result['val_acc']))