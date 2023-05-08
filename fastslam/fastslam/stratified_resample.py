import numpy as np
from .stratified_random import stratified_random

def stratified_resample(w):
    # [keep, Neff] = stratified_resample(w)
    # INPUT:
    #   w - set of N weights [w1, w2, ..]
    # OUTPUTS:
    #   keep - N indices of particles to keep 
    #   Neff - number of effective particles (measure of weight variance)
    
    w = w / sum(w) # normalise
    Neff = 1 / sum(w ** 2)
    
    l = len(w)
    keep = np.zeros(l)
    select = stratified_random(l)
    w = np.cumsum(w)
    
    ctr = 0
    for i in range(l):
        while ctr < l and select[ctr] < w[i]:
            keep[ctr] = i
            ctr += 1
    keep = keep.astype(int)
            
    return keep, Neff
