from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from dataLoader import getDataset
import torch



#-----------------------------------------------------------------------------
#           Current Standard - Weighted Sum
#-----------------------------------------------------------------------------
'''
Current Standard:
    * Weighted sum of inputs to output
    
'''

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1278, 1),
            
            )
        

    def forward(self, x):
        return self.net(x)


net = BaseNet()

#-----------------------------------------------------------------------------
#           Define Helper Functions
#-----------------------------------------------------------------------------

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        count = 0
        correct = 0
        total = 0
        for i, data in enumerate(testData, 0):           
            count +=1
            # Get inputs
            inputs, targets = data
            print("targets", targets.shape)
            print("inputs ", inputs.shape)
            # Generate outputs
            outputs = network(inputs.float())
            print("output", outputs.shape)
            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            print("Predictions", predicted.shape)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if count > 50:
                break
        # Print accuracy
        print('Accuracy: %d %%' % (100 * correct / total))

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []

    if opt_func == torch.optim.SGD:
        optimizer = opt_func(model.parameters(), lr, momentum=0.9)
    else:
        optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plotResults(history,name):
    losses = [entry['val_loss'] for entry in history]
    accuracy = [entry["val_acc"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Model Results')

    ax1.plot(losses, '-o', label="Validation Loss")
    ax1.plot(train_loss, "-s", label="Training Loss")
    ax1.legend()
    ax1.set_ylim([0,5])
    ax1.set(xlabel = 'Epoch', ylabel="Loss")

    
    ax2.set(xlabel = 'Epoch', ylabel="Values")
    ax2.plot(accuracy, "-r")

    # plt.legend()
    ax1.set_title('Loss vs. Number of Epochs');
    ax2.set_title("Top 1% Accuracy on Validation Set");
    plt.savefig("{}-results.png".format(name))
    plt.show()

    
histories = []

#-----------------------------------------------------------------------------
# Load the train and test data
#-----------------------------------------------------------------------------
data = getDataset()
dataset = DataLoader(data, batch_size=64, num_workers = 1)
train_loader = dataset
val_loader = dataset
testData = DataLoader(data, batch_size=64, num_workers = 1)

device = get_default_device()
network = BaseNet()
network = network.float()

def main():
    model = to_device(BaseNet(), device)
    history = [evaluate(model, val_loader)]
    num_epochs = 10
    opt_func = torch.optim.SGD
    lr = 4e-2
    history += fit(num_epochs, lr, model, train_loader, val_loader)
    histories.append(history)

if __name__ == '__main__':
    main()


