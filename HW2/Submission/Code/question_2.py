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
from backward_selection_drop_worst import BackwardSelectionDropWorst
from pipeline import Pipeline
import matplotlib.pyplot as plt
from grid_search_cv import GridSearchCV_new
import time
from sklearn.svm import SVR
import itertools as it
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV

from sklearn.feature_selection import SelectFromModel



def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))

def get_data(path):


    path = '../../Data/BlogFeedback/'
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_distribute.npy')
    train_x = train[:, 0:train.shape[1] - 1]
    train_y = train[:, -1]
    test_x = test[:, 0:test.shape[1] - 1]


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


def two_point_one():
    path = 'BlogFeedback'
    train_x, train_y, test_x = get_data(path)

    classifier = Ridge()
    rmse_scorer = make_scorer(rmse, greater_is_better=False)


    ##########################################
    # This code takes 15h to run. That is why I commented it out
    # It will produce the features in best_features
    ##########################################
    # feature_selection = BackwardSelection(train_x)
    # feature_params = None
    # tuned_parameters =    {'alpha': [14000]}
    # clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=2)
    # steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}
    # pipe = Pipeline(steps, feature_params)
    # pipe.fit(train_x, train_y)
    # pipe.print_best()
    # prediction = pipe.predict(train_x)


# BEST PARAMS FROM pipe
#######################################
    second_best =  [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 31, 32, 33, 36, 37, 39, 40, 46, 51, 54, 55, 57, 62, 65, 70, 71, 76, 84, 87, 91, 94, 103, 105, 106, 107, 112, 115, 122, 127, 128, 130, 132, 133, 135, 136, 145, 146, 147, 151, 163, 170, 172, 174, 177, 180, 184, 188, 191, 195, 200, 203, 206, 207, 210, 213, 214, 218, 219, 223, 224, 226, 228, 234, 236, 240, 241, 243, 246, 248, 251, 252, 255, 259, 261, 262, 263, 266, 267, 271, 272, 274, 275, 277]
    best_features = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 31, 32, 33, 36, 37,
                     39, 40, 46, 51, 54, 55, 57, 62, 65, 70, 71, 76, 84, 87, 91, 94, 103, 105, 106, 107, 112, 115, 122,
                     127, 128, 130, 132, 133, 135, 136, 145, 146, 147, 151, 163, 170, 172, 174, 177, 180, 184, 188, 191,
                     195, 200, 203, 206, 207, 210, 213, 214, 218, 219, 223, 224, 226, 228, 234, 236, 240, 241, 243, 246,
                     248, 251, 252, 255, 259, 261, 262, 263, 266, 267, 271, 272, 274, 275, 277]

#######################################



    tuned_parameters = {'alpha': np.arange(5000,7000,100.0),
                        'tol': np.arange(0.00,0.1,0.01)
                        }

    best_params = {'alpha': 6100.0, 'tol': 0.0}
    # score = 29.4503547388
    # kaggle = 24.47664

    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=2)

    train_x = train_x[:, best_features]
    test_x = test_x[:, best_features]
    clf.fit(train_x,train_y)
    print(clf.best_score_)
    print(clf.best_params_)
    prediction = clf.predict(test_x)

    file = "../Predictions/" + path + "/" + "Ridge_Regression_BackwardSelection" + "_best.csv"
    kaggleize(prediction, file)


def two_point_two():

    path = 'BlogFeedback'
    train_x, train_y, test_x = get_data(path)
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    classifier = KNeighborsRegressor()
    tuned_parameters = {'n_neighbors': range(22,26)}
    lasso = LassoCV()
    feature_selection = SelectFromModel(lasso)
    feature_params = {'threshold' : np.arange(0.05,0.07,0.01)}
    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)
    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}
    pipe = Pipeline(steps, feature_params)
    pipe.fit(train_x, train_y)
    pipe.print_best()
    prediction = pipe.predict(test_x)

    #Fitting Finished
    # ###########################
    # best score = 25.734737594
    # best features = {'threshold': 0.044999999999999998}
    # best parameters = {'n_neighbors': 24}
    # ###########################

    file = "../Predictions/" + path + "/" + "best.csv"
    kaggleize(prediction, file)

# two_point_one()
#two_point_two()

