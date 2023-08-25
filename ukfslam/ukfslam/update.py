import numpy
from .observe_model import observe_model
from .unscented_update import unscented_update
from .pi_to_pi import pi_to_pi

def update(XX, PX, z,R,idf):
    for i in range(len(idf)):
        XX,PX = unscented_update(observe_model, observediff, XX,PX, z[:,[i]],R, idf[i])
    return XX, PX

def observediff(z1, z2):
    dz = z1-z2
    dz[1,:] = pi_to_pi(dz[1,:])
    return dz 
