import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import copy
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA





class MyPipeline():
    params = {}
    def __init__(self, elems):
        self.elems = elems
        if not self.__check_correctness__():
            raise ValueError("Pipe is not correct")
        self.__loop_elem__()


    def __check_correctness__(self):

        for i in range(0,len(self.elems)-1):
            if not hasattr(self.elems[i][1], "fit_transform"):
                return False
        if hasattr(self.elems[-1][1], "fit_transform"):
            return False
        return True


    def __fit_transform__(self, elems, train_x, train_y):

        if(len(elems) == 1):
            self.params[elems[0][0]] =elems[0][1].get_params()
            return self.predict(elems[0][1], train_x,train_y)
        else:
            #try:
            elem = elems[0][1]
            self.params[elems[0][0]] = elem.get_params()
            elems.pop(0)

            #test = elem.fit_transform(train_x,  train_y)
            return self.__fit_transform__(elems, elem.fit_transform(train_x, train_y), train_y)

            #except:
                #elem = elems[0][1]
                #elems.pop(0)
                #test = elem.fit_transform(train_x)
                #self.params[elems[0][0]] = elem.get_params()
                #return self.__fit_transform__(elems, elem.fit_transform(train_x), train_y)



    def predict(self, elem, train_x, train_y):
        print "bla"

    def score(self):
        print "bla"

    def __loop_elem__(self):

        train_x, train_y, test_y = get_data('AirFoil')
        self.__fit_transform__(copy.copy(self.elems), train_x, train_y)



        print "bla"

    def fit(self):
        print "bla"

    def get_params(self, deep):
        return self.params

    def set_params(self):
        print "bla"


def adaBoost_final_optimization_email():
    path = 'Email_spam'
    #train_x, train_y, test_x = get_data(path)
    #train_x = [1,2,3]
    #train_y = [1,1,0]
    train_x, train_y, test_y = get_data('AirFoil')

    classifier = AdaBoostClassifier()

    tuned_parameters = [{'classifier__base_estimator': [RandomForestClassifier()],
                         'classifier__base_estimator__bootstrap': [True],
                         'classifier__base_estimator__class_weight': [None],
                         'classifier__base_estimator__criterion': ['gini', 'entropy'],  # 2
                         'classifier__base_estimator__max_depth': range(3, 12, 3),  # 4
                         'classifier__base_estimator__max_features': ['auto'],
                         # 'classifier__base_estimator__max_leaf_nodes': [1],
                         'classifier__base_estimator__min_samples_leaf': [1, 2, 3],  # 3
                         'classifier__base_estimator__min_samples_split': [1, 2, 3, 4],  # 4
                         'classifier__base_estimator__warm_start': [False],
                         'classifier__n_estimators': range(1, 20, 5),  # 4
                         'classifier__learning_rate': [1.0],  # range(0.5,1.0,0.1),
                         'classifier__algorithm': ['SAMME', 'SAMME.R']  # 2

                         }]





    pca = PCA()
    pipe = MyPipeline([('pca', pca), ('classifier', classifier)])

    pipe2 = Pipeline([('pca', pca), ('classifier', classifier)])

    clf = GridSearchCV(pipe, param_grid=tuned_parameters, scoring='accuracy', cv=10)
    clf.fit(train_x, train_y)


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

adaBoost_final_optimization_email()
