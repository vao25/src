from scipy.linalg import block_diag
import numpy as np
from .unscented_transform import unscented_transform

def augment(XX, PX, z, R):
    # add new features to state
    for i in range(z.shape[1]):
        XX, PX = add_one_z(XX, PX, z[:,i], np.copy(R))
    return XX, PX

def add_one_z(XX, PX, z, R):
    XX = np.vstack((XX, np.array([[z[0]],[z[1]]])))
    PX = block_diag(PX, R)
    XX,PX = unscented_transform(augment_model, None, XX,PX)
    return XX, PX

def augment_model(x):
    phi = np.copy(x[2, :])
    r = np.copy(x[-2, :])
    b = np.copy(x[-1, :])
    s = np.sin(phi + b) 
    c = np.cos(phi + b)

    x[-2, :] = x[0,:] + r*c
    x[-1, :] = x[1,:] + r*s 
    return x
