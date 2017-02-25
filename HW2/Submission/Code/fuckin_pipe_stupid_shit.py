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
from pipeline import Pipeline
import matplotlib.pyplot as plt

import itertools as it

def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))

def get_data(path):
    train = np.load("../../Data/" + path + '/train.npy')
    try:
        test = np.load("../../Data/" + path + '/test_distribute.npy')
    except:
        test = np.load("../../Data/" + path + '/test_private.npy')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    test_x = test[:, 1:]
    test_y = test[:, 0]

    return train_x, train_y, test_x, test_y


def plot_line_graph(arrays, labels, title_img, colors = ['ro-', 'bo-']):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(arrays)):
        array = arrays[i]
        index = range(len(array))
        values = array
        ax.plot(values, colors[i])
        ax.set_xlabel('hyperparameter')
        ax.set_ylabel('score')
    plt.title(title_img)

    #plt.legend(labels, loc = 'center')

    plt.savefig("../Figures/"+title_img+".pdf")





def optimize():

    train_x, train_y, test_x , test_y = get_data('AirFoil')

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


    feature_selection = PCA()
    feature_params = {'n_components': range(1,6)  }

    feature_params = {'score_func': [f_regression], 'k': range(1,6) }
    feature_selection = SelectKBest()

    feature_selection = ForwardSelection(train_x)
    feature_params = None

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter = 1000,scoring=rmse_scorer, cv = 10)

    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}

    pipe = Pipeline(steps, feature_params)
    pipe.fit(train_x, train_y)
    pipe.print_best()

    plot_line_graph([pipe.get_all_scores()], ["Parameters"], "RMSE Score" ,colors = ['ro-'])

    prediction = pipe.predict(test_x)
    print "done"

#optimize()



def one_point_six():
    classifier = Ridge()
    train_x, train_y, test_x, test_y = get_data('AirFoil')
    classifier.fit(train_x,train_y)
    prediction = classifier.predict(test_x)
    score = rmse(prediction, test_y)

    print("\n##############################################################")
    print("Default Ridge Classification RMSE Score = "+ str(score))
    print("##############################################################\n")





one_point_six()



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

