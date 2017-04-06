import numpy as np
from sklearn.metrics import accuracy_score

class cluster_class:
    
    def __init__(self,K):
        '''
        Create a cluster classifier object
        '''
        self.K=K #

    def fit(self, X,Y):
        '''
        Learn a cluster classifier object
        '''        
        pass
        
    def predict(self, X):
        '''
        Make predictions usins a cluster classifier object
        '''        
        return np.zeros((X.shape[0],))
    
    def score(self,X,Y):
        '''
        Compute prediction error rate for a cluster classifier object
        '''          
        Yhat = self.predict(X)
        return 1-accuracy_score(Y,Yhat)
        