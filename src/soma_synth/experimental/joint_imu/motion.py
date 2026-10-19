"""Checkpoint-compatible research architecture; see JOINT_IMU.md for provenance."""
import math,torch
from torch import nn
from torch.nn import functional as F
from .pose import PoseCorrection,skew
def expso(v):
 a=v.norm(dim=-1);k=skew(v);return torch.eye(3,device=v.device)+torch.sinc(a/torch.pi)[...,None,None]*k+.5*torch.sinc(a/(2*torch.pi))[...,None,None]**2*(k@k)
class Block(nn.Module):
 def __init__(self,w,d):super().__init__();self.conv=nn.Conv1d(w,w,3,padding=d,dilation=d);self.norm=nn.GroupNorm(8,w)
 def forward(self,x):return x+F.silu(self.norm(self.conv(x)))
class Body(nn.Module):
 def __init__(self,kind,width):
  super().__init__();self.kind=kind;self.proj=nn.Conv1d(336,width,1)
  if kind=='tcn':self.net=nn.Sequential(*[Block(width,d) for d in [1,2,4,8,16,32]])
  elif kind=='attention':self.net=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*3,dropout=0,batch_first=True,activation='gelu'),2)
  else:self.down=nn.Conv1d(width,width,5,stride=2,padding=2);self.down2=nn.Conv1d(width,width,5,stride=2,padding=2);self.net=nn.Sequential(Block(width,1),Block(width,2),Block(width,4));self.up=nn.Conv1d(width*2,width,3,padding=1)
 def forward(self,x):
  x=F.silu(self.proj(x))
  if self.kind=='tcn':return self.net(x)
  if self.kind=='attention':
   t=torch.arange(x.shape[-1],device=x.device)[:,None];f=torch.exp(torch.arange(0,x.shape[1],2,device=x.device)*(-math.log(10000)/x.shape[1]));pe=torch.zeros(x.shape[-1],x.shape[1],device=x.device);pe[:,0::2]=torch.sin(t*f);pe[:,1::2]=torch.cos(t*f)
   return self.net(x.transpose(1,2)+pe).transpose(1,2)
  h=F.silu(self.down(x));z=self.net(F.silu(self.down2(h)));z=F.interpolate(z,size=x.shape[-1],mode='linear',align_corners=False)
  return self.up(torch.cat([x,z],1))
class Synth(nn.Module):
 def __init__(self,cfg,mean,scale):
  super().__init__();self.cfg=cfg;self.register_buffer('mean',torch.tensor(mean)[None,:,None]);self.register_buffer('scale',torch.tensor(scale)[None,:,None]);self.penalty_value=None
  if cfg['kind']=='baseline':self.base=PoseCorrection(mean,scale)
  else:
   self.body=Body(cfg['arch'],cfg['width']);self.head=nn.Conv1d(cfg['width'],64,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x,m):
  if self.cfg['kind']=='baseline':
   out=x+self.base((x-self.mean)/self.scale)*self.scale;self.penalty_value=self.base.regularization();return out
  b,_,t=x.shape
  # Marker position, velocity and acceleration plus pairwise geometry/quality.
  raw=m.reshape(b,8,3,3,t).permute(0,1,4,2,3)*.1
  a=raw[...,1,:]-raw[...,0,:];c=raw[...,2,:]-raw[...,0,:]
  edges=torch.stack([a.norm(dim=-1),c.norm(dim=-1),(c-a).norm(dim=-1)],-1)
  delta=edges-edges.median(dim=2,keepdim=True).values
  quality=torch.cat([delta/.003,torch.gradient(delta,spacing=(.01,),dim=(2,))[0]/.1,edges/.1],-1).permute(0,1,3,2).reshape(b,72,t)
  #72quality channels replace raw acceleration if disabled, maintaining parameter count.
  velocity=torch.gradient(m,spacing=(.01,),dim=(2,))[0]*.1
  high=torch.gradient(velocity,spacing=(.01,),dim=(2,))[0]/20
  q=quality if self.cfg['quality'] else torch.zeros_like(quality)
  # Same 336-channel architecture in both arms. Control masks only the
  # previously omitted channels; treatment exposes all marker accelerations.
  acc_features=high.clamp(-20,20)
  if not self.cfg['all_acceleration']:
   acc_features=torch.cat([acc_features[:,:24],torch.zeros_like(acc_features[:,24:])],1)
  inp=torch.cat([(x-self.mean)/self.scale,m,velocity.clamp(-20,20),q.clamp(-20,20),acc_features],1)
  assert inp.shape[1]==336
  h=self.head(self.body(inp));field=h[:,:48].reshape(b,2,8,3,t).permute(0,1,2,4,3)
  if self.cfg['kind']=='direct':
   correction=self.cfg['cap']*torch.tanh(h[:,:48])*self.scale;self.penalty_value=.0001*(correction/self.scale).square().mean();return x+correction
  gates=torch.sigmoid(h[:,48:].reshape(b,2,8,t,1)) if self.cfg['kind']=='gated' else 1.
  if self.cfg.get('constant_gate',False):
   gates=torch.sigmoid(self.head.bias[48:]).reshape(1,2,8,1,1)
  field=torch.tanh(field)*gates;d=self.cfg['cap']*.005*field[:,0];theta=self.cfg['cap']*.034906585*field[:,1]
  derivative=lambda u:torch.gradient(u,spacing=(.01,),dim=(2,))[0]
  f=x[:,:24].reshape(b,8,3,t).permute(0,1,3,2);w=x[:,24:].reshape(b,8,3,t).permute(0,1,3,2)
  dot=derivative(d);acc=derivative(dot)+2*torch.cross(w,dot,dim=-1)+torch.cross(derivative(w),d,dim=-1)+torch.cross(w,torch.cross(w,d,dim=-1),dim=-1)
  rot=expso(theta);qt=rot.transpose(-1,-2);om=qt@derivative(rot);om=.5*(om-om.transpose(-1,-2));dw=torch.stack([om[...,2,1],om[...,0,2],om[...,1,0]],-1)
  fo=(qt@(f+acc)[...,None]).squeeze(-1);wo=(qt@w[...,None]).squeeze(-1)+dw
  # Observation penalty is a geometric proxy, not a calibrated marker likelihood.
  ex=F.normalize(a,dim=-1);ez=F.normalize(torch.cross(a,c,dim=-1),dim=-1);ey=torch.cross(ez,ex,dim=-1);r=torch.stack([ex,ey,ez],-1)
  local=(raw-raw.mean(-2,keepdim=True))@r
  moved=(local@rot.transpose(-1,-2))-local+d[...,None,:]
  variance=delta.square().mean(-1).detach().clamp(.0005**2,.003**2)
  obs=F.smooth_l1_loss(moved/variance.sqrt()[...,None,None],torch.zeros_like(moved),reduction='none').mean()
  self.penalty_value=self.cfg['observation']*obs+.0001*(field.square().mean())+.001*(field.diff(dim=3).square().mean())
  return torch.cat([fo.permute(0,1,3,2).reshape(b,24,t),wo.permute(0,1,3,2).reshape(b,24,t)],1)
