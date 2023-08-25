import numpy as np

# angle = pi_to_pi(angle)
# Input: array of angles.
def pi_to_pi(angle):
    angle = np.mod(angle, 2*np.pi)
    i = np.where(angle > np.pi)
    angle[i] = angle[i] - 2*np.pi
    i = np.where(angle < -np.pi)
    angle[i] = angle[i] + 2*np.pi
    return angle
 
