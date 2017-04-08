import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import random
from sklearn.linear_model import LogisticRegression


class cluster_class_bonus:
    def __init__(self, K):
        '''
        Create a cluster classifier object
        '''
        self.cluster = KMeans(n_clusters=K, random_state=10)
        self.classifier = LogisticRegression()

        self.pred_list = [0]*K
        self.K = K  #

    def fit(self, X, Y):
        '''
        Learn a cluster classifier object
        '''

        self.cluster.fit(X)
        labels = self.cluster.labels_
        self.classifier.fit(labels.reshape((-1,1)),Y)



    def predict(self, X):
        '''
        Make predictions usins a cluster classifier object
        '''

        labels = self.cluster.predict(X)
        labels = self.classifier.predict(labels.reshape((-1,1)))

        return labels

    def score(self, X, Y):
        '''
        Compute prediction error rate for a cluster classifier object
        '''
        Yhat = self.predict(X)
        return 1 - accuracy_score(Y, Yhat)







        # import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.cluster import KMeans
# import cluster_utils
#
#
# class cluster_class:
#
#     def __init__(self,K, rand_state):
#         '''
#         Create a cluster classifier object
#         '''
#         self.rand_state = rand_state
#         self.K=K #
#         self.classifier = KMeans(n_clusters=self.K, random_state=self.rand_state)
#
#
#     def fit(self, X,Y):
#         '''
#         Learn a cluster classifier object
#         '''
#         return self.classifier.fit(X,Y)
#
#     def predict(self, X):
#         '''
#         Make predictions usins a cluster classifier object
#         '''
#         return self.classifier.predict(X)
#
#     def score(self,X,Y):
#         '''
#         Compute prediction error rate for a cluster classifier object
#         '''
#         Yhat = self.predict(X)
#         p = cluster_utils.cluster_proportions(Yhat, self.K)
#         test = cluster_utils.cluster_means(X,Yhat,self.K)
#         return cluster_utils.cluster_quality(X,Yhat,self.K)
#         #return 1-accuracy_score(Y,Yhat)
#