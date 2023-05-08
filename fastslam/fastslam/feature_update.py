import numpy as np
from .pi_to_pi import pi_to_pi
from .compute_jacobians import compute_jacobians
from .KF_cholesky_update import KF_cholesky_update
from .KF_joseph_update import KF_joseph_update

def feature_update(particle, z, idf, R):
    # particle= feature_update(particle, z, idf, R)
    # Having selected a new pose from the proposal distribution, this pose is assumed
    # perfect and each feature update may be computed independently and without pose uncertainty.
    
    if idf.size != 0:
        xf = np.copy(particle.xf[:,idf])
        Pf = np.copy(particle.Pf[:,:,idf])
    
        zp, Hv, Hf, Sf = compute_jacobians(particle, idf, R)
        v = z - zp
        v[1,:] = pi_to_pi(v[1,:])
    
    for i in range(len(idf)):
        vi = np.copy(v[:,i])
        Hfi = np.copy(Hf[:,:,i])
        Pfi = np.copy(Pf[:,:,i])
        xfi = np.copy(xf[:,i])
        
        xf[:,i], Pf[:,:,i] = KF_cholesky_update(xfi, Pfi, vi, R, Hfi)
        
    if idf.size != 0:
        particle.xf[:,idf] = xf
        particle.Pf[:,:,idf] = Pf
    
    return particle 
