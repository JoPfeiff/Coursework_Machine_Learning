import numpy as np
    
def computeXY(rdd):    
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''
    return np.random.randn(41,)

def computeXX(rdd):
    '''
    Compute the outer product term
    needed by OLS. Return as a numpy array
    of size (41,41)
    '''    
    return np.random.randn(41,41)
    
def computeWeights(rdd):  
    '''
    Compute the linear regression weights.
    Return as a numpy array of shape (41,)
    '''
    return np.random.randn(41,)
    
def computePredictions(w,rdd):  
    '''
    Compute predictions given by the input weight vector
    w. Return an RDD with one prediction per row.
    '''
    return rdd
    
def computeError(w,rdd):
    '''
    Compute the MAE of the predictions.
    Return as a float.
    '''
    return np.inf