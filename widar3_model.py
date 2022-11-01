import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce




class Widar3_LSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar3_LSTM,self).__init__()
        self.lstm = nn.LSTM(30,32,num_layers=1)
        self.fc = nn.Linear(32,num_classes)
    def forward(self,x):
        x = x.permute(1, 0, 2)
        _, (ht,ct) = self.lstm(x)
        #ht[-1] --> 64 * 64
        outputs = self.fc(ht[-1])
        return outputs
