import torch
from torch import nn

def zero_head(head):
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

class ResidualTCN(nn.Module):
    def __init__(self,outputs=48,dilations=(1,2,4,8)):
        super().__init__();layers=[];c=48
        for d in dilations:layers.extend([nn.Conv1d(c,64,3,padding=d,dilation=d),nn.ReLU()]);c=64
        self.net=nn.Sequential(*layers,nn.Conv1d(64,outputs,1));zero_head(self.net[-1])
    def forward(self,x):return self.net(x)


