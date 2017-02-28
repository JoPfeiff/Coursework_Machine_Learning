import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
import copy
#from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import make_scorer
# import forward_selection as for_selec
from forward_selection import ForwardSelection
import matplotlib.pyplot as plt

import time

import itertools as it



class Pipeline():

    def __init__(self, steps, feature_params = None):
        self.best = {}
        self.best['score'] = -float('inf')
        self.best['features'] = None
        self.best['parameters'] = None
        self.best['estimator'] = None
        self.scores = []
        self.X = None
        self.Y = None
        try:
            self.feature_optimizer = steps['feature_optimizer']
            self.hyper_optimizer = steps['hyper_optimizer']
            self.feature_params = feature_params
        except:
            print "Wrong params"

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.feature_params is not None:
            all_feature_params = self.generate_parameters(self.feature_params)
            for features in all_feature_params:
                self.feature_optimizer.set_params(**features)
                X_current = self.feature_optimizer.fit_transform(X,Y)
                self.hyper_optimizer.fit(X_current,Y)
                score = self.hyper_optimizer.best_score_
                if(self.score_comparer(score, self.best['score'])):
                    self.best['score'] = score
                    self.best['features'] = features
                    self.best['parameters'] = self.hyper_optimizer.best_params_
                    self.best['estimator'] = self.hyper_optimizer.best_estimator_

        else:
            while True:
                start = time.time()
                X_current = self.feature_optimizer.fit_transform()
                if X_current is False:
                    break
                self.hyper_optimizer.fit(X_current, Y)
                score = self.hyper_optimizer.best_score_
                self.feature_optimizer.set_score(score)
                grid_Scores = self.hyper_optimizer.grid_scores_

                for elem in grid_Scores:
                    self.scores.append(-elem[1])

                if (self.score_comparer(score, self.best['score'])):
                    self.best['score'] = score
                    self.best['features'] = copy.copy(self.feature_optimizer.get_best_params())
                    self.best['parameters'] = copy.copy(self.hyper_optimizer.best_params_)
                    self.best['estimator'] = copy.copy(self.hyper_optimizer.best_estimator_)

                end = time.time()

                print("one iteration = %s seconds, current heap is %s long") %(end - start, len(self.feature_optimizer.heap) )



        print "Fitting Finished"

    def predict(self, X):
        if self.feature_params is not None:
            self.feature_optimizer.set_params(**self.best['features'])
        #self.best['estimator'].
        X_current = self.feature_optimizer.transform(X)
        try:
            prediction = self.best['estimator'].predict(X_current)
        except:
            prediction = self.fit_predict(X)
        return prediction

    def fit_predict(self, test_x):
        train_x = self.feature_optimizer.transform(self.X)
        test_x = self.feature_optimizer.transform(self.Y)
        classifier = self.best['estimator'].fit(train_x, self.Y)
        # classifier.set_params(**self.best['parameters'])
        # classifier.fit(train_x,train_y)
        prediction = classifier.predict(test_x)
        return prediction

    def get_all_scores(self):
        return self.scores

    def score_comparer(self, score1 , score2):
        return score1 > score2


    def print_best(self):
        print("###########################")
        print "best score = " + str(-self.best['score'] )
        print "best features = " + str(self.best['features'] )
        print "best parameters = " + str(self.best['parameters'])
        print("###########################")

    def generate_parameters(self, all_params):
        iterables = [all_params[param] for param in all_params]
        # then also generate all parameter definitions
        keys = [param for param in all_params]
        # then generate a dictionary with all combinations
        return [{keys[i]: tup[i] for i in range(0, len(keys))} for tup in it.product(*iterables)]


