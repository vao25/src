import numpy as np
from math import sin, cos

def add_feature(particle, z, R):
    # add new features
    lenz = z.shape[1]
    xf = np.zeros((2,lenz))
    Pf = np.zeros((2,2,lenz))
    xv = particle.xv
    
    for i in range(lenz):
        r = z[0,i]
        b = z[1,i]
        y = xv[2] + b
        s = sin(y[0])
        c = cos(y[0])
        
        e = xv[0] + r*c
        f = xv[1] + r*s
        xf[:,i] = np.array([e[0], f[0]])
        
        Gz = np.array([[c, -r*s], [s,  r*c]])
        Pf[:,:,i] = np.dot(np.dot(Gz, R), Gz.T)
        
    #lenx = particle.xf.shape[1]
    #ii = np.arange(lenz) + (lenx)
    particle.xf = np.append(particle.xf, xf, axis = 1)
    particle.Pf = np.append(particle.Pf, Pf, axis = 2)
    return particle
