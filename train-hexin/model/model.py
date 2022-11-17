import torch
import torch.nn as nn

class HeXin3_LSTM(nn.Module):
    def __init__(self,num_classes):
        super(HeXin3_LSTM,self).__init__()
        self.lstm = nn.LSTM(4,32,num_layers=1)
        self.fc = nn.Linear(32,num_classes)
    def forward(self,x):
        try:
            x=x.permute(1, 0, 2)
        except:
            pass
        _, (ht,ct) = self.lstm(x)
        #ht[-1] --> 64 * 64
        outputs = self.fc(ht[-1])
        return outputs