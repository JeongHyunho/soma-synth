"""Checkpoint-compatible research architecture; see JOINT_IMU.md for provenance."""
import torch
from torch import nn
from .legacy_layers import ResidualTCN,PhysicsDisplacement

def skew(v):
 x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
 return torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*v.shape[:-1],3,3)

class PoseCorrection(nn.Module):
 """Effective local trajectory correction; no measured IMU inference inputs.
 Q maps corrected sensor axes to baseline sensor axes.
 Relative-motion identities use baseline specific force and body angular rate.
 Finite differences approximate continuous derivatives; not independent pose supervision.
 """
 def __init__(self,mean,scale):
  super().__init__();self.net=ResidualTCN(48,(1,2,4))
  self.register_buffer('mean',torch.as_tensor(mean)[None,:,None]);self.register_buffer('scale',torch.as_tensor(scale)[None,:,None])
 def fields(self,x):
  b,_,t=x.shape;z=self.net(x).reshape(b,2,8,3,t).permute(0,1,2,4,3)
  return .01*torch.tanh(z[:,0]),(.0872664626)*torch.tanh(z[:,1])
 def forward(self,x):
  b,_,t=x.shape;physical=x*self.scale+self.mean
  f=physical[:,:24].reshape(b,8,3,t).permute(0,1,3,2)
  w=physical[:,24:].reshape(b,8,3,t).permute(0,1,3,2)
  d,theta=self.fields(x);self.last_fields=(d,theta)
  derivative=lambda v:torch.gradient(v,spacing=(.01,),dim=(2,),edge_order=1)[0]
  dot=derivative(d);delta=derivative(dot)+2*torch.cross(w,dot,dim=-1)+torch.cross(derivative(w),d,dim=-1)+torch.cross(w,torch.cross(w,d,dim=-1),dim=-1)
  # Stable Rodrigues formula, including a differentiable zero-angle limit.
  angle=torch.linalg.vector_norm(theta,dim=-1);K=skew(theta)
  Q=torch.eye(3,device=x.device,dtype=x.dtype)+torch.sinc(angle/torch.pi)[...,None,None]*K+.5*torch.sinc(angle/(2*torch.pi))[...,None,None]**2*(K@K)
  QT=Q.transpose(-1,-2);H=QT@derivative(Q);H=.5*(H-H.transpose(-1,-2))
  relative_w=torch.stack([H[...,2,1],H[...,0,2],H[...,1,0]],-1)
  fc=(QT@(f+delta)[...,None]).squeeze(-1);wc=(QT@w[...,None]).squeeze(-1)+relative_w
  correction=torch.cat([(fc-f).permute(0,1,3,2).reshape(b,24,t),(wc-w).permute(0,1,3,2).reshape(b,24,t)],1)
  return correction/self.scale
 def regularization(self):
  d,r=self.last_fields;d=d[:,:,15:-15]/.01;r=r[:,:,15:-15]/.0872664626
  return .0001*(d.square().mean()+r.square().mean())+.001*(d.diff(dim=2).square().mean()+r.diff(dim=2).square().mean())
