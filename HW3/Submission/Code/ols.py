import numpy as np



def get_denom(rdd, min):
    max = rdd.reduce(lambda x,y: np.maximum(x,y))
    return   max - min

def get_min(rdd):
    #return rdd.reduce(lambda x,y: np.amin(x,y))
    return rdd.reduce(lambda x,y: np.minimum(x,y))
    # return rdd.reduce(lambda x,y: )


def normalize(rdd, denom, min):
    return rdd.map(lambda r: (r - min) / denom )

def computeXY(rdd):    
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''

    # min = get_min(rdd)
    # max = get_max(rdd)
    # denom = get_max(rdd) - min
    # test = rdd.map(lambda r: (r - min) / denom).collect
    # return rdd.map(lambda r: (r - min) / denom)
    min = get_min(rdd)
    denom = get_denom(rdd, min )
    rdd = normalize(rdd, denom, min)
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