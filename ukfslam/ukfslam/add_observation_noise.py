import numpy as np

def add_observation_noise(z,R, addnoise):
    # z = add_observation_noise(z,R, addnoise)
    # Add random measurement noise. We assume R is diagonal.
    
    if addnoise == 1:
        l= z.shape[1]
        if l > 0:
            z[0,:]= z[0,:] + np.random.randn(1,l)*np.sqrt(R[0,0])
            z[1,:]= z[1,:] + np.random.randn(1,l)*np.sqrt(R[1,1])
    return z


