import numpy as np
from .pi_to_pi import pi_to_pi

def vehicle_model(x, V, G, WB, dt):
    xv = np.array([x[0,:] + V*dt*np.cos(G + x[2,:]),
          x[1,:] + V*dt*np.sin(G + x[2,:]),
          x[2,:] + V*dt*np.sin(G)/WB])
    xv[2,:] = pi_to_pi(xv[2,:])
    return xv
