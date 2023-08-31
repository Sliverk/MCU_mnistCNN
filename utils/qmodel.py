import torch
import torch.nn as nn
import torch.nn.functional as F

class QMnistModel(nn.Module):
    def __init__(self):
        super(QMnistModel, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1,6,5,1,2)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.conv3 = nn.Conv2d(16,120,5,1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2d(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.relu3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dequant(x)
        output =  F.log_softmax(x, dim=1)
        return output