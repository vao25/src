import numpy as np
from .pi_to_pi import pi_to_pi
from .observe_model import observe_model

def data_associate(x, P, z, R, gate1, gate2):
    """
    Simple gated nearest-neighbour data-association. No clever feature
    caching tricks to speed up association, so computation is O(N), where
    N is the number of features in the state.
    """ 

    zf = np.array([[],[]])
    idf = np.array([])
    zn = np.array([[],[]])

    Nxv = 3 # number of vehicle pose states
    Nf = (len(x) - Nxv) / 2 # number of features already in map
    Nf = int(Nf)
    
    # linear search for nearest-neighbour
    for i in range(z.shape[1]):
        jbest = -1
        nbest = float('inf')
        outer = float('inf')
    
        # search for neighbours
        for j in range(Nf):
            nis, nd = compute_association(x, P, np.array([[z[0,i]],[z[1,i]]]), R, j)
            if (nis < gate1) and (nd < nbest): # if within gate, store nearest-neighbour
                nbest = nd
                jbest = j
            elif nis < outer: # else store best nis value
                outer = nis
        
        #  add nearest-neighbour to association list
        if jbest != -1:
            zf = np.append(zf, [[z[0,i]], [z[1,i]]], axis = 1)
            idf = np.append(idf, jbest)
        elif outer > gate2: # z too far to associate, but far enough to be a new feature
            zn = np.append(zn, [[z[0,i]], [z[1,i]]], axis = 1)
    
    if idf.size != 0:
        idf = idf.astype(int)
    return zf, idf, zn

def compute_association(x, P, z, R, idf):
    # return normalised innovation squared (ie, Mahalanobis distance) and normalised distance
    zp, H = observe_model(x, idf)
    v = z - zp
    v[1] = pi_to_pi(v[1])
    S = np.dot(np.dot(H, P), H.T) + R
    
    nis = np.dot(np.dot(v.T, np.linalg.inv(S)), v)
    nd = nis + np.log(np.linalg.det(S))
    
    return nis, nd
