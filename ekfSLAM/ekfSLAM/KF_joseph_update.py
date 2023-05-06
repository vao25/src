import numpy as np

def KF_joseph_update(x,P,v,R,H):
    # [x,P]= KF_joseph_update(x,P,v,R,H)
    # This module is identical to KF_simple_update() except that
    # it uses the Joseph-form covariance update, as shown in 
    # Bar-Shalom "Estimation with Applications...", 2001, p302.
    
    PHt= np.dot(P,H.T)
    S= np.dot(H,PHt) + R
    Si= np.linalg.inv(S)
    Si= make_symmetric(Si)
    PSD_check= np.linalg.cholesky(Si)
    W= np.dot(PHt,Si)

    x= x + np.dot(W,v) 

    # Joseph-form covariance update
    C= np.eye(P.shape[0]) - np.dot(W,H)
    P= np.dot(np.dot(C,P),C.T) + np.dot(np.dot(W,R),W.T)

    P= P + np.eye(P.shape[0])*np.spacing(1) # a little numerical safety
    PSD_check= np.linalg.cholesky(P)
    return x,P

def make_symmetric(P):
    P= (P+P.T)*0.5
    return P
