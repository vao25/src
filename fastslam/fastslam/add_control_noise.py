import numpy as np
from .multivariate_gauss import multivariate_gauss

def add_control_noise(V,G,Q, addnoise):
    # Add random noise to nominal control values
    if addnoise == 1:
        C= multivariate_gauss(np.array([[V],[G]]),Q, 1) # if Q might be correlated
        V= C[0]
        G= C[1]
    return V[0],G[0]
