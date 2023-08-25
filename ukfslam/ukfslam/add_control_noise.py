import numpy as np
import math

def add_control_noise(V,G,Q, addnoise):
    #[V,G]= add_control_noise(V,G,Q, addnoise)
    #
    # Add random noise to nominal control values. We assume Q is diagonal.

    if addnoise == 1:
        V= V + np.random.randn(1)*math.sqrt(Q[0,0])
        G= G + np.random.randn(1)*math.sqrt(Q[1,1])
        return V[0],G[0]
    return V,G
