from .pi_to_pi import pi_to_pi
from .multivariate_gauss import  multivariate_gauss
import numpy as np
from math import cos
from math import sin

def predict(particle, V, G, Q, WB, dt, addrandom):
    # add random noise to controls
    if addrandom == 1:
        VG = multivariate_gauss(np.array([[V], [G]]), Q, 1)
        V = VG[0]
        G = VG[1]
    
    # predict state
    xv = particle.xv
    particle.xv = np.array([xv[0] + V*dt*cos(G + xv[2]), xv[1] + V*dt*sin(G + xv[2]), pi_to_pi(xv[2] + V*dt*sin(G)/WB)])
    return particle
