import numpy as np

def multivariate_gauss(x, P, n):
    """
    s= multivariate_gauss(x,P,n)

     INPUTS: 
       (x, P) mean vector and covariance matrix
       obtain n samples
     OUTPUT:
       sample set

     Random sample from multivariate Gaussian distribution.
    """
    l = len(x)
    S = np.linalg.cholesky(P).T
    X = np.random.randn(l, n)
    s = np.dot(S, X) + np.dot(x, np.ones((1, n)))
    return s

