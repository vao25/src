import numpy as np
from .pi_to_pi import pi_to_pi
from .observe_model import observe_model
from .KF_cholesky_update import KF_cholesky_update
from .KF_joseph_update import KF_joseph_update

def update(x,P,z,R,idf, batch):
    """
    [x,P]= update(x,P,z,R,idf, batch)

    Inputs:
    x, P - SLAM state and covariance
    z, R - range-bearing measurements and covariances
    idf - feature index for each z
    batch - switch to specify whether to process measurements together or sequentially

    Outputs:
    x, P - updated state and covariance
    """
    
    if batch == 1:
        x,P= batch_update(x,P,z,R,idf)
    else:
        x,P= single_update(x,P,z,R,idf)
    return x,P

#
#

def batch_update(x,P,z,R,idf):
    lenz= z.shape[1]
    lenx= len(x)
    H= np.zeros((2*lenz, lenx))
    v= np.zeros((2*lenz, 1))
    RR= np.zeros((2*lenz,2*lenz))
    
    for i in range(lenz):
        i1= 2*i
        i2= 2*i + 2
        zp,H[i1:i2,:]= observe_model(x, idf[i])
    
        v[i1:i2]=      np.array([[z[0,i]-zp[0][0]],
                              pi_to_pi(z[1,i]-zp[1])])
        RR[i1:i2,i1:i2]= np.copy(R)
    
    x,P= KF_joseph_update(x,P,v,RR,H)
    return x,P

#
#

def single_update(x,P,z,R,idf):
    lenz= z.shape[1]
    for i in range(lenz):
        zp,H= observe_model(x, idf[i])
    
        v= np.array([[z[0,i]-zp[0][0]],
                     pi_to_pi(z[1,i]-zp[1])])
    
        x,P= KF_joseph_update(x,P,v,R,H)
    return x,P
