import numpy as np
from .compute_jacobians import compute_jacobians
from .pi_to_pi import pi_to_pi

def compute_weight(particle, z, idf, R):
    zp, Hv, Hf, Sf = compute_jacobians(particle, idf, R)
    v = z - zp
    v[1,:] = pi_to_pi(v[1,:])
    w = 1
    for i in range(z.shape[1]):
        S = Sf[:,:,i]
        den = 2 * np.pi * np.sqrt(np.linalg.det(S))
        num = np.exp(-0.5 * v[:,i].T.dot(np.linalg.inv(S)).dot(v[:,i]))
        w = w * num / den
    return w
 
