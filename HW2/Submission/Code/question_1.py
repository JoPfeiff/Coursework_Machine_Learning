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
from backward_selection import BackwardSelection
from pipeline import Pipeline
import matplotlib.pyplot as plt
from grid_search_cv import GridSearchCV_new

import itertools as it

def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))

def score_comparer(score1, score2):
    return score1 > score2


def get_data(path):
    path = '../../Data/AirFoil/'
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_private.npy')
    train_x = train[:, 0:train.shape[1] - 1]
    train_y = train[:, -1]
    test_x = test[:, 0:test.shape[1] - 1]
    test_y = test[:, -1]
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




def one_point_six():
    classifier = Ridge()
    train_x, train_y, test_x, test_y = get_data('AirFoil')
    classifier.fit(train_x,train_y)
    prediction = classifier.predict(test_x)
    score = rmse(prediction, test_y)

    print("\n##############################################################")
    print("Question 1.6: Default Ridge Classification RMSE Score = "+ str(score))
    print("##############################################################\n")


def one_point_seven():
    classifier = Ridge()
    train_x, train_y, test_x, test_y = get_data('AirFoil')
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    #Optimizing
    tuned_parameters1 = {'alpha': np.arange(0.00, .0101, 0.001),
                        'copy_X': [True, False],
                        'fit_intercept': [True, False],
                        #'max_iter': range(800, 1600, 200),
                        'normalize': [True],
                        'solver': ['sag'],
                        'tol': np.arange(0.0, 0.1, 0.01)
                        }

    clf1 = GridSearchCV(classifier, param_grid=tuned_parameters1, scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter=1000, scoring=rmse_scorer, cv=10)
    clf1.fit(copy.deepcopy(train_x),copy.deepcopy(train_y))
    print clf1.best_estimator_
    print clf1.best_score_



    ###################################################
    # Graph for different alpha values
    ###################################################
    x_ticks2 = np.arange(0.00, 0.1, 0.001)
    tuned_parameters2 = {'normalize': [True],
                        'solver': ['sag'],
                        'fit_intercept': [True],
                        'tol': [0.05],
                        'copy_X': [False],
                        'alpha': x_ticks2}

    classifier2 = copy.deepcopy(classifier)

    clf2 = GridSearchCV(classifier2, param_grid=tuned_parameters2, scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter=1000, scoring=rmse_scorer, cv=10)

    clf2.fit(copy.deepcopy(train_x), copy.deepcopy(train_y))
    grid_Scores2 = clf2.grid_scores_

    #cv_scores = clf.

    train_scores2 = []
    test_scores2 = []
    all_scores2 = []
    all_scores2.append(train_scores2)
    all_scores2.append(test_scores2)
    for elem2 in grid_Scores2:
        param2 = elem2[0]
        train_score2 = elem2[1]
        classifier2.set_params(**param2)
        classifier2.fit(copy.deepcopy(train_x),copy.deepcopy(train_y))
        prediction2 = classifier2.predict(test_x)
        test_score2 = rmse(prediction2,copy.deepcopy(test_y))
        all_scores2[0].append(abs(train_score2))
        all_scores2[1].append(abs(test_score2))

    plot_line_graph(all_scores2, ["Validation Scores", "Test Scores"], "RMSE Score - Alpha",x_ticks2, "Alpha" )




    ###################################################
    # Graph for different tol values
    ###################################################
    x_ticks3 = np.arange(0.00, 0.2, 0.001)
    tuned_parameters3 = {'normalize': [True],
                        'solver': ['sag'],
                        'fit_intercept': [True],
                        'tol': x_ticks3,
                        'copy_X': [False],
                        'alpha': [0.005]}
    classifier3 =  copy.deepcopy(classifier)
    clf3 = GridSearchCV(classifier, param_grid=tuned_parameters3, scoring=rmse_scorer, cv=10)
    #clf = RandomizedSearchCV(classifier, param_distributions=tuned_parameters, n_iter=1000, scoring=rmse_scorer, cv=10)

    clf3.fit(copy.deepcopy(train_x), copy.deepcopy(train_y))
    grid_Scores3 = clf3.grid_scores_

    train_scores3 = []
    test_scores3 = []
    all_scores3 = []
    all_scores3.append(train_scores3)
    all_scores3.append(test_scores3)
    for elem3 in grid_Scores3:
        param3 = elem3[0]
        train_score3 = elem3[1]
        classifier3.set_params(**param3)
        classifier3.fit(copy.deepcopy(train_x),copy.deepcopy(train_y))
        prediction3 = classifier3.predict(copy.deepcopy(test_x))
        test_score3 = rmse(prediction3,test_y)
        all_scores3[0].append(abs(train_score3))
        all_scores3[1].append(abs(test_score3))

    plot_line_graph(all_scores3, ["Validation Scores", "Test Scores"], "RMSE Score - Tol",x_ticks3, "Tol" )


    #################################################
    # Predict with best parameters
    #################################################
    best_parameters4 = {'normalize': True,
                            'solver': 'sag',
                            'fit_intercept': True,
                            'tol': 0.05, #'tol': 0.069999999999999993
                            'copy_X': False,
                            'alpha': 0.005}#0.04} # 4.7880535399
    classifier4 = copy.deepcopy(classifier)
    classifier4.set_params(**best_parameters4)
    classifier4.fit(copy.deepcopy(train_x),copy.deepcopy(train_y))
    prediction4 = classifier4.predict(copy.deepcopy(test_x))
    best_score4 = rmse(prediction4,copy.deepcopy(test_y))
    print("########################################################")
    print ("Question 1.7: Best RMSE Score Hyper Parameter Selection: %s") %(best_score4)
    print("########################################################")




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
    print("Question 1.8: The Following Features were chosen: %s") %(best['features'])
    print ("Question 1.8: Best RMSE Score Parameter Selection: %s") % (best_score)
    print("########################################################")



def one_point_nine():

    train_x, train_y, test_x , test_y = get_data('AirFoil')



    classifier = Ridge()

    tuned_parameters = {'alpha': np.arange(0.00,0.01,0.001),
                        # 'copy_X': [True,False],
                        # 'fit_intercept': [True,False],
                        #'max_iter': range(800,1600,200),
                        'normalize': [True],
                        'solver': ['sag'],
                        'tol': np.arange(0.00,0.1,0.001)
                        #'random_state':
                        }



    feature_selection = BackwardSelection(train_x)
    #feature_selection = BackwardSelectionDropWorst(train_x)
    feature_params = None

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    clf = GridSearchCV(classifier, param_grid=tuned_parameters,scoring=rmse_scorer, cv=10)

    steps = {'feature_optimizer': feature_selection, 'hyper_optimizer': clf}

    pipe = Pipeline(steps, feature_params)
    pipe.fit(train_x, train_y)
    pipe.print_best()

    #plot_line_graph([pipe.get_all_scores()], ["Parameters"], "RMSE Score" ,colors = ['ro-'])

    prediction = pipe.predict(test_x)
    best_score = rmse(prediction,test_y)
    print("########################################################")
    print ("Question 1.9: Best RMSE Score Pipe: %s") %(best_score)
    print("########################################################")

#optimize()





one_point_six()
one_point_seven()
one_point_eight()
one_point_nine()



