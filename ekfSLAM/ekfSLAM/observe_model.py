import numpy as np

def observe_model(x, idf):
    """
    [z,H]= observe_model(x, idf)

     INPUTS:
       x - state vector
       idf - index of feature order in state

     OUTPUTS:
       z - predicted observation
       H - observation Jacobian

    Given a feature index (ie, the order of the feature in the state vector), predict the expected range-bearing observation of this feature and its Jacobian.
    """
    
    Nxv = 3 # number of vehicle pose states
    fpos = Nxv + idf*2 # position of xf in state
    H = np.zeros((2, len(x)))

    # auxiliary values
    dx = x[fpos][0]  - x[0][0]
    dy = x[fpos+1][0]- x[1][0]
    d2 = dx**2 + dy**2
    d = np.sqrt(d2)
    xd = dx/d
    yd = dy/d
    xd2 = dx/d2
    yd2 = dy/d2

    # predict z
    z = np.array([[d],
                  np.arctan2(dy,dx) - x[2]])

    # calculate H
    H[:,0:3]        = np.array([[-xd, -yd, 0],
                                [yd2, -xd2, -1]])
    H[:,fpos:fpos+2]= np.array([[xd, yd],
                                [-yd2, xd2]])
    return z, H
