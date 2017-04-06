import numpy as np

def count(rdd):
    '''
    Computes the number of rows in rdd.
    Returns the answer as a float.
    '''
    return float(rdd.map(lambda r: 1).reduce(lambda x,y: x+y))

def mean(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (D,)
    Returns the sample mean of each column of rdd as a numpy array of shape (D,)
    '''
    sums = rdd.reduce(lambda x,y: x+y)
    return sums/float(count(rdd))

def std(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (D,)
    Returns the sample standard deviation of 
    each column of rdd as a numpy array of shape (D,)
    '''
    count_ = float(count(rdd))
    mean_values = mean(rdd)
    return rdd.map(lambda r:  ((r-mean_values)**2)/count_).reduce(lambda x,y: x+y) ** 0.5

   

def dot(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (2,)
    Returns the inner (dot) product between the columns as a float.
    '''

    return rdd.map(lambda r: r[0]*r[1]).reduce(lambda x,y: x+y)

def corr(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (2,)
    Returns the sample Pearson's correlation between the columns as a float.
    '''
    std_ = std(rdd)
    mean_ = mean(rdd)
    cov = rdd.map(lambda r: (r[0]-mean_[0])*(r[1]-mean_[1])).reduce(lambda x,y: x+y)


    return cov/((std_[0]*std_[1])*count(rdd))
    
def distribution(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (1,)
    and that the values in rdd are whole numbers in [0,K] for some K.
    Returns the empirical distribution of the values in rdd
    as an array of shape (K+1,)
    '''


    count_  = float(count(rdd))
    return rdd.map(lambda r: (r, 1) ).reduceByKey(lambda x,y: x+y).map(lambda r: r[1]/count_).collect()
