import numpy as np

def stratified_random(N):
    # s = stratified_random(N)
    # Generate N uniform-random numbers stratified within interval (0,1).
    # The set of samples, s, are in ascending order.
    k = 1/N
    di = np.arange(k/2, (1+k/2), k) # deterministic intervals
    s = di + np.random.uniform(size=N) * k - (k/2) # dither within interval
    return s 
