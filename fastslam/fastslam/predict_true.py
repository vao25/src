from .pi_to_pi import pi_to_pi
from math import cos
from math import sin
import numpy as np

def predict_true(xv, V, G, WB, dt):
    xv = np.copy(xv)
    xv[0] = xv[0] + V*dt*cos(G+xv[2])
    xv[1] = xv[1] + V*dt*sin(G+xv[2])
    xv[2] = pi_to_pi(xv[2] + V*dt*sin(G)/WB)
    return xv
