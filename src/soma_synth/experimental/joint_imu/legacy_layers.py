"""Checkpoint-compatible research architecture; see JOINT_IMU.md for provenance."""
import torch
from torch import nn
import torch.nn.functional as F

def zero_head(head):
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(48,64,1),nn.ReLU(),nn.Conv1d(64,64,1),nn.ReLU(),nn.Conv1d(64,48,1));zero_head(self.net[-1])
    def forward(self,x):return self.net(x)

class ResidualTCN(nn.Module):
    def __init__(self,outputs=48,dilations=(1,2,4,8)):
        super().__init__();layers=[];c=48
        for d in dilations:layers.extend([nn.Conv1d(c,64,3,padding=d,dilation=d),nn.ReLU()]);c=64
        self.net=nn.Sequential(*layers,nn.Conv1d(64,outputs,1));zero_head(self.net[-1])
    def forward(self,x):return self.net(x)

class WindowRNN(nn.Module):
    """Exactly 31 samples per prediction in both training and inference."""
    def __init__(self,kind):
        super().__init__();cls=nn.GRU if kind=='gru' else nn.LSTM
        self.rnn=cls(48,32,batch_first=True,bidirectional=True);self.head=nn.Linear(64,48);zero_head(self.head)
    def forward(self,x):
        b,c,t=x.shape
        windows=F.pad(x,(15,15)).unfold(2,31,1).permute(0,2,3,1).reshape(b*t,31,c)
        pieces=[]
        for z in windows.split(512):
            _,hidden=self.rnn(z)
            if isinstance(hidden,tuple):hidden=hidden[0]
            pieces.append(self.head(torch.cat([hidden[-2],hidden[-1]],dim=1)))
        return torch.cat(pieces).reshape(b,t,48).permute(0,2,1)

class PhysicsDisplacement(nn.Module):
    """Predict bounded sensor-local displacement; differentiate rigid-offset physics.
    Sensor orientation stays fixed relative to its segment, so gyro correction is zero.
    """
    def __init__(self,mean,scale):
        super().__init__();self.net=ResidualTCN(24,(1,2,4))
        self.register_buffer('mean',torch.as_tensor(mean)[None,:,None]);self.register_buffer('scale',torch.as_tensor(scale)[None,:,None])
    def forward(self,x):
        physical=x*self.scale+self.mean;b,_,t=x.shape
        w=physical[:,24:].reshape(b,8,3,t).permute(0,1,3,2)
        d=.01*torch.tanh(self.net(x)).reshape(b,8,3,t).permute(0,1,3,2)
        derivative=lambda z:torch.gradient(z,spacing=(.01,),dim=(2,),edge_order=1)[0]
        dd=derivative(d);alpha=derivative(w)
        correction=derivative(dd)+2*torch.cross(w,dd,dim=-1)+torch.cross(alpha,d,dim=-1)+torch.cross(w,torch.cross(w,d,dim=-1),dim=-1)
        a=correction.permute(0,1,3,2).reshape(b,24,t)/self.scale[:,:24]
        return torch.cat([a,torch.zeros_like(a)],1)

def build(kind,mean,scale):
    if kind=='mlp':return ResidualMLP()
    if kind=='tcn':return ResidualTCN()
    if kind in ['gru','lstm']:return WindowRNN(kind)
    if kind=='physics_displacement':return PhysicsDisplacement(mean,scale)
    raise ValueError(kind)
