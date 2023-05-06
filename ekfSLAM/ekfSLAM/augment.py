import numpy as np
from math import sin, cos

def augment(x,P,z,R):
    """
    [x,P]= augment(x,P,z,R)

     Inputs:
       x, P - SLAM state and covariance
       z, R - range-bearing measurements and covariances, each of a new feature

     Outputs:
       x, P - augmented state and covariance

     Notes: 
       - We assume the number of vehicle pose states is three.
       - Only one value for R is used, as all measurements are assumed to have same noise properties.
    """
    
    # add new features to state
    for i in range(z.shape[1]):
        x,P= add_one_z(x,P,z[:,i],R)
    return x,P

def add_one_z(x,P,z,R):
    l= len(x)
    lP = P.shape[0]
    r= z[0]
    b= z[1]
    s= sin(x[2][0] + b)
    c= cos(x[2][0] + b)
    
    # augment x
    x= np.vstack((x,np.array([[x[0][0] + r*c],[x[1][0] + r*s]])))
    
    # jacobians
    Gv= np.array([[1,0,-r*s],[0,1,r*c]])
    Gz= np.array([[c,-r*s],[s,r*c]])
    
    # augment P
    P = np.vstack((P, np.zeros((2,lP))))
    P = np.hstack((P, np.zeros((lP+2,2))))
    rng= np.arange(l,l+2)
    P[np.ix_(rng,rng)]= Gv.dot(P[0:3,0:3]).dot(Gv.T) + Gz.dot(R).dot(Gz.T) # feature cov
    P[np.ix_(rng,np.arange(3))]= Gv.dot(P[0:3,0:3]) # vehicle to feature xcorr
    P[np.ix_(np.arange(3),rng)]= np.copy(P[np.ix_(rng,np.arange(3))].T)
    if l>3:
        rnm= np.arange(3,l)
        P[np.ix_(rng,rnm)]= Gv.dot(P[0:3,rnm]) # map to feature xcorr
        P[np.ix_(rnm,rng)]= np.copy(P[np.ix_(rng,rnm)].T)
    return x,P
