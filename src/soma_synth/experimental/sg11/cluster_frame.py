"""Three-marker frame; static sensor alignment without fixed triangle dimensions."""
import numpy as np
from scipy.spatial.transform import Rotation

def frame(m):
    finite=np.isfinite(m).all((1,2))&(np.linalg.norm(m,axis=2)>1e-9).all(1)
    a=m[:,1]-m[:,0];b=m[:,2]-m[:,0]
    la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1)
    cross=np.cross(a,b);area=np.linalg.norm(cross,axis=1)
    valid=finite&(la>.005)&(lb>.005)&(area>0.05*la*lb)
    r=np.full((len(m),3,3),np.nan)
    x=a[valid]/la[valid,None];z=cross[valid]/area[valid,None];y=np.cross(z,x)
    r[valid]=np.stack([x,y,z],axis=2)
    # Defined geometric origin: midpoint of marker2/marker3, not a sensor-origin assumption.
    p=(m[:,1]+m[:,2])*.5
    return p,r,valid

def apply(m,cal):
    p,r,v=frame(m);outp=p+np.einsum('tij,j->ti',r,np.asarray(cal['offset']))
    outr=r@np.asarray(cal['alignment']);return outp,outr,v
