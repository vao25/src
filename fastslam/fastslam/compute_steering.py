import numpy as np
import math
from .pi_to_pi import pi_to_pi

def compute_steering(x, wp, iwp, minD, G, rateG, maxG, dt):
    
    """
    INPUTS:
      x - true position
      wp - waypoints
      iwp - index to current waypoint
      minD - minimum distance to current waypoint before switching to next
      G - current steering angle
      rateG - max steering rate (rad/s)
      maxG - max steering angle (rad)
      dt - timestep
 
    OUTPUTS:
      G - new current steering angle
      iwp - new current waypoint
    """
    
    # determine if current waypoint reached
    cwp = np.copy(wp[:,iwp])
    d2 = (cwp[0]-x[0])**2 + (cwp[1]-x[1])**2
    if d2 < minD**2:
        iwp = iwp+1 # switch to next
        if iwp >= wp.shape[1]: # reached final waypoint, flag and return
            iwp = -1
            return G, iwp
        cwp = np.copy(wp[:,iwp]) # next waypoint
    # compute change in G to point towards current waypoint
    deltaG = pi_to_pi(math.atan2(cwp[1]-x[1], cwp[0]-x[0]) - x[2] - G)
    # limit rate
    maxDelta = rateG*dt
    if abs(deltaG) > maxDelta:
        deltaG = np.sign(deltaG)*maxDelta
    # limit angle
    G = G+deltaG
    if abs(G) > maxG:
        G = np.sign(G)*maxG
    G = G[0]
    return G, iwp

