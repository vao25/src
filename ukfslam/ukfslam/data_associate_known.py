import numpy as np

def data_associate_known(x, z, idz, table):
    # zf,idf,zn, table= data_associate_known(x,z,idz, table)
    
    # For simulations with known data-associations, this function maintains
    # a feature/observation lookup table. It returns the updated table, the
    # set of associated observations and the set of observations to new features.
    
    zf = np.array([[],[]])
    idf = np.array([])
    zn = np.array([[],[]])
    idn = np.array([])
    
    # find associations (zf) and new features (zn)
    for i in range(len(idz)):
        ii = idz[i]
        if table[0,ii] == -1: # new feature
            zn = np.append(zn, [[z[0,i]], [z[1,i]]], axis = 1)
            idn = np.append(idn, ii)
        else:
            zf = np.append(zf, [[z[0,i]], [z[1,i]]], axis = 1)
            idf = np.append(idf, table[0,ii])
    
    # add new feature IDs to lookup table        
    Nxv = 3 # number of vehicle pose states
    Nf = (len(x) - Nxv)/2 # number of features already in map
    if idn.size != 0:
        idn = idn.astype(int)
        for j in range(zn.shape[1]):
            table[0,idn[j]] = Nf  + j # add new feature positions to lookup table
            
    if idf.size != 0:
        idf = idf.astype(int)
            
    return zf, idf, zn, table

