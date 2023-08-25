import numpy as np

def observe_model(x, idf):
    """
    z= observe_model(x, idf)

     INPUTS:
       x - state vector
       idf - index of feature order in state

     OUTPUTS:
       z - predicted observation

    Given a feature index (ie, the order of the feature in the state vector), predict the expected range-bearing observation of this feature.
    """
    
    Nxv = 3 # number of vehicle pose states
    fpos = Nxv + idf*2 # position of xf in state

    # auxiliary values
    dx = x[fpos,:]  - x[0,:]
    dy = x[fpos+1,:]- x[1,:]
    d2 = dx**2 + dy**2
    d = np.sqrt(d2)

    # predict z
    z = np.array([d,
                  np.arctan2(dy,dx) - x[2,:]])

    return z
