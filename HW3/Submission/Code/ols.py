import numpy as np


def computeXY(rdd):    
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''

    return rdd.map(lambda r: np.multiply(add_bias(r),r[-1])).reduce(lambda x,y: x+y)

def computeXX(rdd):
    '''
    Compute the outer product term
    needed by OLS. Return as a numpy array
    of size (41,41)
    '''    
    return rdd.map(lambda r: np.outer(add_bias(r),add_bias(r))).reduce(lambda x,y: x+y)
    
def computeWeights(rdd):  
    '''
    Compute the linear regression weights.
    Return as a numpy array of shape (41,)
    '''
    return np.matmul(np.linalg.inv(computeXX(rdd)),computeXY(rdd))
    
def computePredictions(w,rdd):  
    '''
    Compute predictions given by the input weight vector
    w. Return an RDD with one prediction per row.
    '''
    return rdd.map(lambda r: np.dot(w,add_bias(r)))
    
def computeError(w,rdd):
    '''
    Compute the MAE of the predictions.
    Return as a float.
    '''
    count = rdd.map(lambda r: 1).reduce(lambda x,y: x+y)
    rdd = rdd.map(lambda r: (r[-1], np.dot(w,add_bias(r))))
    return rdd.map(lambda r: abs(r[0]-r[1])).reduce(lambda x,y: x+y)/count

def add_bias(array):
    return np.append(array[:-1],1)