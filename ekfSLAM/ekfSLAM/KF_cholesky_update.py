import numpy as np

def KF_cholesky_update(x,P,v,R,H):
    """
    [x,P]= KF_cholesky_update(x,P,v,R,H)

    Calculate the KF (or EKF) update given the prior state [x,P]
    the innovation [v,R] and the (linearised) observation model H.
    The result is calculated using Cholesky factorisation, which
    is more numerically stable than a naive implementation.
    """
    
    PHt = np.dot(P,H.T)
    S = np.dot(H,PHt) + R
    S = (S + S.T)*0.5 # make symmetric
    SChol = np.linalg.cholesky(S)
    SCholInv = np.linalg.inv(SChol) # triangular matrix
    W1 = np.dot(PHt,SCholInv)
    W = np.dot(W1,SCholInv.T)
    x = x + np.dot(W,v) # update
    P = P - np.dot(W1,W1.T)
    return x,P
