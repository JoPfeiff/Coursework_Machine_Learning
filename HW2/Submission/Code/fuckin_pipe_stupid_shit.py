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
import forward_selection as fs

import itertools as it

def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))



class ForwardSelection():

    def __init__(self, training_data):
        self.training_data = training_data
        self.best_params = []
        self.current_params = []
        self.best_score = -float('inf')
        self.betterized = True
        self.heap = []


    def set_score(self, score):
        if score > self.best_score + 0.00000001:
            self.best_params = copy.copy(self.current_params)
            self.best_score = score
            self.betterized = True

    def get_new_heap(self):
        if (len(self.heap) == 0) and not self.betterized:
            return  self.betterized
        elif(len(self.best_params) == self.training_data.shape[1]):
            return False
        else:
            if len(self.heap) == 0:
                self.set_new_heap()
            return self.heap.pop()

    def set_new_heap(self):
        self.betterized = False

        for i in range(0,self.training_data.shape[1]):
            if i not in self.best_params:
                elem = copy.copy(self.best_params)
                elem.append(i)
                self.heap.append(elem)

    def fit_transform(self):
        popped = self.get_new_heap()
        if popped is False:
            return popped
        else:
            self.current_params = popped
            return self.training_data[:, popped]

    def transform(self,data):
        return data[:, self.best_params]


    def get_best_params(self):
        return self.best_params


class Pipeline():

    def __init__(self, steps, feature_params = None):
        self.best = {}
        self.best['score'] = -float('inf')
        self.best['features'] = None
        self.best['parameters'] = None
        self.best['estimator'] = None
        try:
            #self.classifier = steps['classifier']
            self.feature_optimizer = steps['feature_optimizer']
            self.hyper_optimizer = steps['hyper_optimizer']
            self.feature_params = feature_params #params['feature_params']
            #self.hyper_params = params['hyper_params']
            #self.cv = cv
        except:
            print "Wrong params"

    def fit(self, X, Y):

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
                X_current = self.feature_optimizer.fit_transform()
                if X_current is False:
                    break
                self.hyper_optimizer.fit(X_current, Y)
                score = self.hyper_optimizer.best_score_
                self.feature_optimizer.set_score(score)
                if (self.score_comparer(score, self.best['score'])):
                    self.best['score'] = score
                    self.best['features'] = copy.copy(self.feature_optimizer.get_best_params())
                    self.best['parameters'] = copy.copy(self.hyper_optimizer.best_params_)
                    self.best['estimator'] = copy.copy(self.hyper_optimizer.best_estimator_)

        print "Fitting Finished"

    def predict(self, X):
        if self.feature_params is not None:
            self.feature_optimizer.set_params(**self.best['features'])
        X_current = self.feature_optimizer.transform(X)
        return self.best['estimator'].predict(X_current)



    def score_comparer(self, score1 , score2):
        if(score1 > score2):
            return True
        else:
            return False


    def print_best(self):
        print("###########################")
        print "best score = " + str(-self.best['score'] )
        print "best features = " + str(self.best['features'] )
        print "best parameters = " + str(self.best['parameters'])
        #print "best   = " + str(self.best['estimator'])
        print("###########################")



    #def pipe(self, param_optimization, param_feat, hyper_param_optimization, classifier,  hyper_param_features, train_x, train_y):
    #
    #     all_params = self.generate_parameters(param_feat)
    #
    #     score = 0.0
    #
    #     best_hyper = None
    #     best_param = None
    #
    #     for param in all_params:
    #         param_optimization.fit_params(**param)
    #         train_x = param_optimization.fit_transform(train_x, train_y)
    #
    #         hyper = hyper_param_optimization(classifier, hyper_param_features)
    #         hyper.fit(train_x,train_y)
    #         if(hyper.best_score() > score):
    #             score = hyper.best_score()
    #             best_hyper = copy.copy(hyper)
    #             best_param = copy.copy(param)


    def generate_parameters(self, all_params):
        # if (all_params is None):
        #     all_params = self.feature_params
        # first generate all parametersettings
        iterables = [all_params[param] for param in all_params]
        # then also generate all parameter definitions
        keys = [param for param in all_params]
        # then generate a dictionary with all combinations
        return [{keys[i]: tup[i] for i in range(0, len(keys))} for tup in it.product(*iterables)]




def get_data(path):
    # path = 'Data/Email_spam/'
    train = np.load("../../Data/" + path + '/train.npy')
    try:
        test = np.load("../../Data/" + path + '/test_distribute.npy')
    except:
        test = np.load("../../Data/" + path + '/test_private.npy')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    test_x = test[:, 1:]

    return train_x, train_y, test_x



def optimize():

    train_x, train_y, test_x = get_data('AirFoil')

    classifier = LinearRegression()
    tuned_parameters = [{'fit_intercept': [True, False],
                         'normalize': [True, False],
                         'copy_X': [True, False],
                         'n_jobs': [-1],  # 2
                         #'classifier__base_estimator__max_depth': range(3, 12, 3),  # 4
                        }]

    classifier = Ridge()

    tuned_parameters = {'alpha': np.arange(1.0,3.0,0.4),
                        'copy_X': [True,False],
                        'fit_intercept': [True,False],
                        'max_iter': range(800,1600,200),
                        'normalize': [True,False],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                        'tol': np.arange(0.8,1.4,0.2)
                        #'random_state':
                        }

    tuned_parameters = {'alpha': np.arange(1.0,3.0,0.4)#,
                        #'copy_X': [True,False],
                        #'fit_intercept': [True,False],
                        #'max_iter': range(800,1600,200),
                        #'normalize': [True,False],
                        #'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                        #'tol': np.arange(0.8,1.4,0.2)
                        #'random_state':
                        }

    #classifier.fit(train_x,train_y)
    #classifier.predict(test_x)

    feature_selection = PCA()
    feature_params = {'n_components': range(1,6)  }

    feature_params = {'score_func': [f_regression], 'k': range(1,6) }
    feature_selection = SelectKBest()

    # test = [[0]] * train_x.shape[0]
    # #print(train_x.shape())
    # train_x = np.append(train_x,test,1)
    #
    # test = [[0]] * test_x.shape[0]
    # #print(train_x.shape())
    # test_x = np.append(test_x,test,1)
    #
    # for i in range(0, test_x.shape[0]-1):
    #     train_x[i,3] = 0

    #ForwardSelection(train_x)
    feature_selection = ForwardSelection(train_x)
    feature_params = None


    #print(train_x.shape())

    rmse_scorer = make_scorer(rmse, greater_is_better=False)
        # rmse, needs_proba=True)

    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter = 1000,scoring=rmse_scorer, cv = 10)

    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}

    pipe = Pipeline(steps, feature_params)
    pipe.fit(train_x, train_y)
    pipe.print_best()
    prediction = pipe.predict(test_x)
    print "done"

optimize()
    # best estimator :
    # Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=800,
    #       normalize=False, random_state=None, solver='auto',
    #       tol=0.80000000000000004)
    # best features:
    # {'k': 5, 'score_func': < function
    # f_regression
    # at
    # 0x10ae3bcf8 >}
    # best params:
    # {'normalize': False, 'fit_intercept': True, 'solver': 'auto', 'max_iter': 800, 'tol': 0.80000000000000004,
    #  'copy_X': True, 'alpha': 1.0}
    # score:
    # -5817722.18013

