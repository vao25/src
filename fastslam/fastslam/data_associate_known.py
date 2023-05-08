import numpy as np

def data_associate_known(z, idz, table, Nf):
    # [zf,idf,zn, table]= data_associate_known(z, idz, table, Nf)
    # For simulations with known data-associations, this function maintains a feature/observation lookup table. It returns the updated table, the set of associated observations and the set of observations to new features.
    
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
    
    if idn.size != 0:
        # add new feature IDs to lookup table
        idn = idn.astype(int)
        for j in range(zn.shape[1]):
            table[0,idn[j]] = Nf  + j # add new feature positions to lookup table
    
    if idf.size != 0:
        idf = idf.astype(int)

    return zf, idf, zn, table

