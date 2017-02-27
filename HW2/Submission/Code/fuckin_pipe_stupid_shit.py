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
from backward_selection_drop_worst import BackwardSelectionDropWorst
from pipeline import Pipeline
import matplotlib.pyplot as plt
from grid_search_cv import GridSearchCV_new

import itertools as it

def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))

def get_data(path):
    path = '../../Data/AirFoil/'
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_private.npy')
    train_x = train[:, 0:train.shape[1] - 1]
    train_y = train[:, -1]
    test_x = test[:, 0:test.shape[1] - 1]

    # Real test targets/outputs
    test_y = test[:, -1]
    # print "Air Foil:", train_x.shape, train_y.shape, test_x.shape, test_y.shape
    #
    #
    # train = np.load("../../Data/" + path + '/train.npy')
    # try:
    #     test = np.load("../../Data/" + path + '/test_distribute.npy')
    # except:
    #     test = np.load("../../Data/" + path + '/test_private.npy')
    # train_x = train[:, 1:]
    # train_y = train[:, 0]
    # test_x = test[:, 1:]
    # test_y = test[:, 0]

    return train_x, train_y, test_x, test_y


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





def one_point_nine():

    train_x, train_y, test_x , test_y = get_data('AirFoil')

    # classifier = LinearRegression()
    # tuned_parameters = [{'fit_intercept': [True, False],
    #                      'normalize': [True, False],
    #                      'copy_X': [True, False],
    #                      'n_jobs': [-1],  # 2
    #                      #'classifier__base_estimator__max_depth': range(3, 12, 3),  # 4
    #                     }]

    classifier = Ridge()

    tuned_parameters = {'alpha': np.arange(0.00,0.01,0.001),
                        'copy_X': [True,False],
                        'fit_intercept': [True,False],
                        'max_iter': range(800,1600,200),
                        'normalize': [True,False],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                        'tol': np.arange(0.01,0.5,0.01)
                        #'random_state':
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

    feature_selection = ForwardSelection(train_x)
    #feature_selection = BackwardSelectionDropWorst(train_x)
    feature_params = None

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    # clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)
    clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter = 1000,scoring=rmse_scorer, cv = 10)

    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}

    pipe = Pipeline(steps, feature_params)
    pipe.fit(train_x, train_y)
    pipe.print_best()

    #plot_line_graph([pipe.get_all_scores()], ["Parameters"], "RMSE Score" ,colors = ['ro-'])

    prediction = pipe.predict(test_x)
    best_score = rmse(prediction,test_y)
    print("########################################################")
    print ("Best RMSE Score Pipe: %s") %(best_score)
    print("########################################################")

#optimize()



def one_point_six():
    classifier = Ridge()
    train_x, train_y, test_x, test_y = get_data('AirFoil')
    classifier.fit(train_x,train_y)
    prediction = classifier.predict(test_x)
    score = rmse(prediction, test_y)
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    print("\n##############################################################")
    print("Default Ridge Classification RMSE Score = "+ str(score))
    print("##############################################################\n")


def one_point_seven():
    classifier = Ridge()
    train_x, train_y, test_x, test_y = get_data('AirFoil')
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    #Optimizing
    tuned_parameters = {'alpha': np.arange(0.001, .0101, 0.001),
                        'copy_X': [True, False],
                        'fit_intercept': [True, False],
                        #'max_iter': range(800, 1600, 200),
                        'normalize': [True, False],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                        'tol': np.arange(0.1, 1.0, 0.1)
                        # 'random_state':
                        }

    clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter=1000, scoring=rmse_scorer, cv=10)
    #clf.fit(train_x,train_y)
    #print clf.best_estimator_


    ###################################################
    # Graph for different alpha values
    ###################################################
    x_ticks = np.arange(0.00, 0.05, 0.001)
    tuned_parameters = {'normalize': [True],
                        'solver': ['sag'],
                        'fit_intercept': [True],
                        'tol': [0.2],
                        'copy_X': [True],
                        'alpha': x_ticks}



    clf = GridSearchCV(classifier, param_grid=tuned_parameters, scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter=1000, scoring=rmse_scorer, cv=10)

    clf.fit(train_x, train_y)
    grid_Scores = clf.grid_scores_

    train_scores = []
    test_scores = []
    all_scores = []
    all_scores.append(train_scores)
    all_scores.append(test_scores)
    for elem in grid_Scores:
        param = elem[0]
        train_score = elem[1]
        classifier.set_params(**param)
        classifier.fit(train_x,train_y)
        prediction = classifier.predict(test_x)
        test_score = rmse(prediction,test_y)
        all_scores[0].append(abs(train_score))
        all_scores[1].append(abs(test_score))

    plot_line_graph(all_scores, ["Train Scores", "Validation Scores"], "RMSE Score - Alpha",x_ticks, "Alpha" )




    ###################################################
    # Graph for different tol values
    ###################################################
    x_ticks = np.arange(0.00, 0.5, 0.02)
    tuned_parameters = {'normalize': [True],
                        'solver': ['sag'],
                        'fit_intercept': [True],
                        'tol': x_ticks,
                        'copy_X': [True],
                        'alpha': [0.04]}


    clf = GridSearchCV(classifier, param_grid=tuned_parameters, scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter=1000, scoring=rmse_scorer, cv=10)

    clf.fit(train_x, train_y)
    grid_Scores = clf.grid_scores_

    train_scores = []
    test_scores = []
    all_scores = []
    all_scores.append(train_scores)
    all_scores.append(test_scores)
    for elem in grid_Scores:
        param = elem[0]
        train_score = elem[1]
        classifier.set_params(**param)
        classifier.fit(train_x,train_y)
        prediction = classifier.predict(test_x)
        test_score = rmse(prediction,test_y)
        all_scores[0].append(abs(train_score))
        all_scores[1].append(abs(test_score))

    plot_line_graph(all_scores, ["Train Scores", "Validation Scores"], "RMSE Score - Tol",x_ticks, "Tol" )


    #################################################
    # Predict with best parameters
    #################################################
    best_parameters = {'normalize': True,
                            'solver': 'sag',
                            'fit_intercept': True,
                            'tol': 0.2, #'tol': 0.069999999999999993
                            'copy_X': True,
                            'alpha': 0.007}#0.04} # 4.7880535399
    classifier.set_params(**best_parameters)
    classifier.fit(train_x,train_y)
    prediction = classifier.predict(test_x)
    best_score = rmse(prediction,test_y)
    print("########################################################")
    print ("Best RMSE Score Hyper Parameter Selection: %s") %(best_score)
    print("########################################################")


def score_comparer(score1, score2):
    if (score1 > score2):
        return True
    else:
        return False

def one_point_eight():
    best = {}
    best['score'] = -float('inf')
    best['features'] = None
    best['parameters'] = None
    best['estimator'] = None
    scores = []
    train_x, train_y, test_x , test_y = get_data('AirFoil')
    classifier = Ridge()
    tuned_parameters = {'normalize': [False]}
    feature_selection = ForwardSelection(train_x)
    feature_params = None
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter = 1000,scoring=rmse_scorer, cv = 10)
    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}

    while True:
        X_current = feature_selection.fit_transform()
        if X_current is False:
            break
        clf.fit(X_current, train_y)
        score = clf.best_score_
        feature_selection.set_score(score)
        grid_Scores = clf.grid_scores_

        if (score_comparer(score, best['score'])):
            best['score'] = score
            best['features'] = copy.copy(feature_selection.get_best_params())
            best['parameters'] = copy.copy(clf.best_params_)
            best['estimator'] = copy.copy(clf.best_estimator_)

    #plot_line_graph([pipe.get_all_scores()], ["Parameters"], "RMSE Score" ,colors = ['ro-'])


    X_current = feature_selection.transform(test_x)
    prediction = best['estimator'].predict(X_current)
    best_score = rmse(prediction, test_y)
    print("########################################################")
    print("The Following Features were chosen: %s") %(best['features'])
    print ("Best RMSE Score Parameter Selection: %s") % (best_score)
    print("########################################################")




one_point_six()
one_point_seven()
one_point_eight()
one_point_nine()



