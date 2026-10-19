"""Checkpoint-compatible research architecture; see JOINT_IMU.md for provenance."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .motion import Synth as Frozen

class Block(nn.Module):
 def __init__(self,w,d):
  super().__init__();self.conv=nn.Conv1d(w,w,3,padding=d,dilation=d);self.norm=nn.GroupNorm(8,w)
 def forward(self,x):return x+F.silu(self.norm(self.conv(x)))

class Synth(nn.Module):
 def __init__(self,cfg,mean,scale,base_config,base_mean,base_scale,sd):
  super().__init__();self.cfg=cfg
  self.base=Frozen(base_config,base_mean,base_scale);self.base.requires_grad_(False);self.base.eval()
  self.register_buffer('sd',torch.as_tensor(sd)[None,:,None]);self.register_buffer('center',torch.tensor(mean)[None,:,None]);self.register_buffer('scale',torch.tensor(scale)[None,:,None])
  self.embedding=nn.Embedding(8,8)
  # Each site consumes its own 6D synthetic signals and 3-marker kinematics.
  # Context arm additionally receives all-site baseline signals, never measured targets.
  dim=53+(48 if cfg['family']=='context' else 0)
  self.proj=nn.Conv1d(dim,64,1);self.body=nn.Sequential(*[Block(64,d) for d in [1,2,4,8,16,32]])
  if cfg['family']=='multiscale':self.slow=nn.Sequential(Block(64,1),Block(64,2),Block(64,4));self.fuse=nn.Conv1d(128,64,1)
  self.out=nn.Conv1d(64,6,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
  self.uncertainty_head=nn.Conv1d(64,6,1);nn.init.zeros_(self.uncertainty_head.weight);nn.init.zeros_(self.uncertainty_head.bias)
  self.teacher=copy.deepcopy(self.base);self.teacher.requires_grad_(False);self.teacher.eval()
  self.base.requires_grad_(bool(cfg['joint']))
 def forward(self,x,m):
  self.base.eval()
  with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg['joint']):base=self.base(x,m)
  self.teacher.eval()
  with torch.no_grad():self.baseline_output=self.teacher(x,m)
  b,_,t=x.shape
  def site(a):return torch.cat([a[:,:24].reshape(b,8,3,t),a[:,24:].reshape(b,8,3,t)],2)
  pos=m.reshape(b,8,9,t);vel=torch.gradient(pos,spacing=(.01,),dim=(3,))[0]*.1;acc=torch.gradient(vel,spacing=(.01,),dim=(3,))[0]/20
  emb=self.embedding(torch.arange(8,device=x.device))[None,:,:,None].expand(b,-1,-1,t)
  z=[site((x-self.center)/self.scale),site((base-self.center)/self.scale),pos,vel.clamp(-20,20),acc.clamp(-20,20),site((base-x)/self.sd),emb]
  if self.cfg['family']=='context':z.append(((base-self.center)/self.scale)[:,None].expand(-1,8,-1,-1))
  h=F.silu(self.proj(torch.cat(z,2).reshape(b*8,-1,t)));fast=self.body(h)
  if self.cfg['family']=='multiscale':
   slow=self.slow(F.avg_pool1d(h,4,4));fast=F.silu(self.fuse(torch.cat([fast,F.interpolate(slow,size=t,mode='linear',align_corners=False)],1)))
  lv=self.uncertainty_head(fast).reshape(b,8,6,t)
  lv=torch.cat([lv[:,:,:3].reshape(b,24,t),lv[:,:,3:].reshape(b,24,t)],1)
  self.log_variance=4*torch.tanh(lv/4) if self.cfg['probabilistic'] else lv*0
  d=self.cfg['residual_cap']*torch.tanh(self.out(fast)).reshape(b,8,6,t)
  d=torch.cat([d[:,:,:3].reshape(b,24,t),d[:,:,3:].reshape(b,24,t)],1)
  self.penalty_value=.001*d.square().mean()+.001*d.diff(dim=-1).square().mean()+self.base.penalty_value
  return base+d*self.sd
