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
from forward_selection import ForwardSelection
from backward_selection import BackwardSelection
from pipeline import Pipeline
import matplotlib.pyplot as plt
from grid_search_cv import GridSearchCV_new
import time

import itertools as it

def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))

def get_data(path):
    # train = np.load("../../Data/" + path + '/train.npy')
    # test = np.load("../../Data/" + path + '/test_distribute.npy')
    # train_x = train[:, 1:]
    # train_y = train[:, 0]
    # test_x = test[:, :]
    # #test_y = test[:, 0]

    path = '../../Data/BlogFeedback/'
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_distribute.npy')
    train_x = train[:, 0:train.shape[1] - 1]
    train_y = train[:, -1]
    test_x = test[:, 0:test.shape[1] - 1]
    # this is dummy vector of predictions
    #test_y = test[:, -1]
    #print "Blog Feedback:", train_x.shape, train_y.shape, test_x.shape, test_y.shape


    return train_x, train_y, test_x



def kaggleize(predictions, file):
    if (len(predictions.shape) == 1):
        predictions.shape = [predictions.shape[0], 1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids, predictions)).astype(float)

    np.savetxt(file, kaggle_predictions, fmt="%f", delimiter=",", header="Id,Prediction")


def plot_line_graph(arrays, labels, title_img, x_ticks, tuning_parameter, colors = ['ro-', 'bo-']):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ticks = np.arange(0.0, 0.5, 0.02)
    for i in range(len(arrays)):
        array = arrays[i]
        index = range(len(array))
        values = array
        ax.plot(x_ticks, values, colors[i])
        ax.set_xlabel(tuning_parameter)
        ax.set_ylabel('RMSE Score')
    plt.title(title_img)
    plt.legend(labels, loc = 'right')

    plt.savefig("../Figures/"+title_img+".pdf")


#best parameters = {'normalize': True, 'tol': 0.069999999999999993, 'fit_intercept': True, 'copy_X': False, 'alpha': 0.0070000000000000001, 'solver': 'sag', 'max_iter': 1200}



def two_point_one():
    path = 'BlogFeedback'
    train_x, train_y, test_x = get_data(path)

    # classifier = LinearRegression()
    # tuned_parameters = [{'fit_intercept': [True, False],
    #                      'normalize': [True, False],
    #                      'copy_X': [True, False],
    #                      'n_jobs': [-1],  # 2
    #                      #'classifier__base_estimator__max_depth': range(3, 12, 3),  # 4
    #                     }]

    classifier = Ridge()

    tuned_parameters = {'alpha': [8900],#np.arange(8800,9001,100.0),
                        'copy_X': [True],
                        'fit_intercept': [False],
                        #'max_iter': range(1000),
                        'normalize': [True],
                        #'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                         'tol': [0.1]#np.arange(0.1,0.31,0.1)
                        # 'random_state':
                        }

    # tuned_parameters = {'alpha': np.arange(1.0,3.0,0.4)#,
    #                     #'copy_X': [True,False],
    #                     #'fit_intercept': [True,False],
    #                     #'max_iter': range(800,1600,200),
    #                     #'normalize': [True,False],
    #                     #'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
    #                     #'tol': np.arange(0.8,1.4,0.2)
    #                     #'random_state':
    #                     }


    # feature_selection = PCA()
    # feature_params = {'n_components': range(1,6)  }
    #
    # feature_params = {'score_func': [f_regression], 'k': range(1,6) }
    # feature_selection = SelectKBest()

    # train_x = train_x[:, [9, 14, 19, 16, 1, 123,137,128,45,56]]
    # test_x = test_x[:, [9, 14, 19, 16, 1, 123,137,128,45,56]]


    # feature_selection = ForwardSelection(train_x)
    feature_selection = BackwardSelection(train_x)
    feature_params = None

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)
    # clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter = 1000,scoring=rmse_scorer, cv = 10)

    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}

    # best
    # score = 1.8521369528
    # best
    # feature_selection.best_params = [9, 14, 19, 16, 1]

    # parameters = {'alpha': 0.0}
    # classifier.set_params(**parameters)
    pipe = Pipeline(steps, feature_params)
    # clf.fit(train_x,train_y)

    # pipe.best['estimator'] = classifier
    pipe.fit(train_x, train_y)
    pipe.print_best()

    #plot_line_graph([pipe.get_all_scores()], ["Parameters"], "RMSE Score" ,colors = ['ro-'])

    prediction = pipe.predict(test_x)
    file = "../Predictions/" + path + "/" + "Ridge_Regression" + "_best.csv"
    kaggleize(prediction, file)
    # best_score = rmse(prediction,test_y)
    # print("########################################################")
    # print ("Best RMSE Score Pipe: %s") %(best_score)
    # print("########################################################")


two_point_one()
#optimize()

