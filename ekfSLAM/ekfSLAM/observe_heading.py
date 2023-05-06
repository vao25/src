import numpy as np
from math import pi
from .pi_to_pi import pi_to_pi
from .KF_joseph_update import KF_joseph_update

def observe_heading(x,P, phi, useheading):
    #function [x,P]= observe_heading(x,P, phi, useheading)
    #
    # Perform state update for a given heading measurement, phi,
    # with fixed measurement noise: sigmaPhi

    if useheading==0: return x,P
    sigmaPhi= 0.01*pi/180 # radians, heading uncertainty

    H= np.zeros(len(x))
    H[2]= 1
    v= pi_to_pi(phi - x[2])[0]

    [x,P]= KF_joseph_update(x,P,v, sigmaPhi**2,H)
    return x,P

