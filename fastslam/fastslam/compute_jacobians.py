import numpy as np
from .pi_to_pi import pi_to_pi

def compute_jacobians(particle, idf, R):
    zp = None
    Hv = None
    Hf = None
    Sf = None
    if idf.size != 0:
        xv = particle.xv
        xf = particle.xf[:,idf]
        Pf = particle.Pf[:,:,idf]
        
        zp = np.zeros((2,len(idf)))
        Hv = np.zeros((2,3,len(idf)))
        Hf = np.zeros((2,2,len(idf)))
        Sf = np.zeros((2,2,len(idf)))
    
    for i in range(len(idf)):
        dx = xf[0,i] - xv[0]
        dy = xf[1,i] - xv[1]
        dx = dx[0]
        dy = dy[0]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        a = pi_to_pi(np.arctan2(dy,dx) - xv[2])
        a = a[0]
        
        zp[:,i] = np.array([d, a]) # predicted observation
        
        Hv[:,:,i] = np.array([[-dx/d, -dy/d, 0], # Jacobian wrt vehicle states
                              [dy/d2, -dx/d2, -1]])
        Hf[:,:,i] = np.array([[dx/d, dy/d], # Jacobian wrt feature states
                              [-dy/d2, dx/d2]])
        Sf[:,:,i] = np.dot(Hf[:,:,i], np.dot(Pf[:,:,i], Hf[:,:,i].T)) + R # innovation covariance of 'feature observation given the vehicle'
        
    return zp, Hv, Hf, Sf
