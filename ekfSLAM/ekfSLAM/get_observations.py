import numpy as np

def get_observations(x, lm, idf, rmax):
    """
        [z,idf]= get_observations(x, lm, idf, rmax)

     INPUTS:
       x - vehicle pose [x;y;phi]
       lm - set of all landmarks
       idf - index tags for each landmark
       rmax - maximum range of range-bearing sensor 

     OUTPUTS:
       z - set of range-bearing observations
       idf - landmark index tag for each observation
    """

    def get_visible_landmarks(x, lm, idf, rmax):
        # Select set of landmarks that are visible within vehicle's semi-circular field-of-view
        dx = np.copy(lm[0, :] - x[0])
        dy = np.copy(lm[1, :] - x[1])
        phi = x[2]
        # incremental tests for bounding semi-circle
        # bounding box, bounding line, bounding circle
        ii = np.where((np.abs(dx) < rmax) & (np.abs(dy) < rmax) & ((dx * np.cos(phi) + dy * np.sin(phi)) > 0) & ((dx ** 2 + dy ** 2) < rmax ** 2))[0]
        # Note: the bounding box test is unnecessary but illustrates a possible speedup technique as it quickly eliminates distant points. Ordering the landmark set would make this operation O(logN) rather that O(N).
        lmC = np.copy(lm[:, ii])
        idf = np.copy(idf[ii])
        return lmC, idf

    def compute_range_bearing(x, lm):
        # Compute exact observation
        dx = np.copy(lm[0, :] - x[0])
        dy = np.copy(lm[1, :] - x[1])
        phi = x[2]
        z = np.array([np.sqrt(dx ** 2 + dy ** 2), np.arctan2(dy, dx) - phi])
        return z
    
    lm, idf = get_visible_landmarks(x, lm, idf, rmax)
    z = compute_range_bearing(x, lm)
    return z, idf
