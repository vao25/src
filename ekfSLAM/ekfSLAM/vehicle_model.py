import numpy as np
from math import sin, cos
from .pi_to_pi import pi_to_pi

def vehicle_model(xv, V, G, WB, dt):
    """
     INPUTS:
       xv - vehicle pose [x;y;phi]
       V - velocity
       G - steer angle (gamma)
       WB - wheelbase
       dt - change in time

     OUTPUTS:
       xv - new vehicle pose
    """
    
    xv = np.array([[xv[0][0] + V*dt*cos(G+xv[2][0])], 
                   [xv[1][0] + V*dt*sin(G+xv[2][0])],
                   pi_to_pi(xv[2] + V*dt*sin(G)/WB)])
    return xv
