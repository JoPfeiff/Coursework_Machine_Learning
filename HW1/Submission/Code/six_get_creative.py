import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def get_data(path):

    train = np.load("../../Data/" + path + '/train.npy')
    test = np.load("../../Data/" + path + '/test_distribute.npy')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    test_x = test[:, 1:]
    
    return train_x, train_y, test_x



def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions)).astype(int)
    writer = csv.writer(open(file, 'w'))
    writer.writerow(['# id','Prediction'])
    writer.writerows(kaggle_predictions)


def adaBoost_final_optimization_email():
    path = 'Email_spam'
    train_x, train_y, test_x = get_data(path)
    
    classifier = AdaBoostClassifier()
    
    tuned_parameters = [{  'classifier__base_estimator': [  RandomForestClassifier()],
                         'classifier__base_estimator__bootstrap': [True], 
                         'classifier__base_estimator__class_weight': [None], 
                         'classifier__base_estimator__criterion': ['gini', 'entropy'], #2
                         'classifier__base_estimator__max_depth': range(3,12,3), #4
                         'classifier__base_estimator__max_features': ['auto'], 
                         #'classifier__base_estimator__max_leaf_nodes': [1],
                         'classifier__base_estimator__min_samples_leaf': [1,2,3], #3
                         'classifier__base_estimator__min_samples_split': [1,2,3,4],#4
                         'classifier__base_estimator__warm_start':[False],
                         'classifier__n_estimators': range(1,20,5), #4
                         'classifier__learning_rate': [1.0],#range(0.5,1.0,0.1),
                         'classifier__algorithm' : ['SAMME', 'SAMME.R']#2

        }  ]

    pipe = Pipeline(steps=[
                           ('classifier', classifier)])
    clf = GridSearchCV(pipe, param_grid=tuned_parameters, scoring='accuracy', cv = 10)
    
    # For fitting again, these need to be uncommented but takes a lot of time
    
    #clf.fit(train_x, train_y)
    #print clf.best_score_
    #print clf.best_estimator_
    
    # best obtained parameter
    clf_best_params = AdaBoostClassifier(algorithm='SAMME.R',
            base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=2, min_samples_split=4,
            warm_start=False),
            learning_rate=1.0, n_estimators=16, random_state=None)
    #0.955322669608
    

    clf_best_params.fit(train_x, train_y)

    scores = cross_val_score(clf_best_params, train_x, train_y, cv=10)

    final_prediction = clf_best_params.predict(test_x)

    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    kaggleize(final_prediction,"../Predictions/"+path+"/Email_AdaBoost_Final_Optimization_Best.csv")
        


#adaBoost_final_optimization_email()



def adaBoost_final_optimization_occupancy():
    path = 'Occupancy_detection'
    train_x, train_y, test_x = get_data(path)
    
    classifier = AdaBoostClassifier()


    tuned_parameters = [{  'classifier__base_estimator': [  RandomForestClassifier()],
                         'classifier__base_estimator__bootstrap': [True], 
                         'classifier__base_estimator__class_weight': [None], 
                         'classifier__base_estimator__criterion': ['gini', 'entropy'],
                         'classifier__base_estimator__max_depth': range(3,12,3), 
                         'classifier__base_estimator__max_features': ['auto'], 
                         #'classifier__base_estimator__max_leaf_nodes': [1],
                         'classifier__base_estimator__min_samples_leaf': [1,2,], 
                         'classifier__base_estimator__min_samples_split': [1,2,3,],
                         'classifier__base_estimator__warm_start':[False],
                         'classifier__n_estimators': range(15,21,5),
                         'classifier__learning_rate': range(1,3,1),#range(0.5,1.0,0.1),
                         'classifier__algorithm' : ['SAMME', 'SAMME.R']

        }            
    ]


    pipe = Pipeline(steps=[('classifier', classifier)])
    clf = GridSearchCV(pipe, param_grid=tuned_parameters, scoring='accuracy', cv = 10)
    
    # For fitting again, these need to be uncommented but takes a lot of time
    #clf.fit(train_x, train_y)
    #print clf.best_score_
    #print  clf.best_estimator_
    
    
    
    clf_best_params = AdaBoostClassifier(algorithm='SAMME',
          base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            warm_start=False),
          learning_rate=1.0, n_estimators=15, random_state=None)#Best Classification
    
    clf_best_params2 = AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
           max_depth=3, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=3,
    #        mi...e=0,
            warm_start=False),
          learning_rate=1, n_estimators=15, random_state=None)
    

    clf_best_params.fit(train_x, train_y)

    scores = cross_val_score(clf_best_params, train_x, train_y, cv=10)

    final_prediction = clf_best_params.predict(test_x)

    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    kaggleize(final_prediction,"../Predictions/"+path+"/Occupancy_detection_AdaBoost_Final_Optimization_Best.csv")






#adaBoost_final_optimization_occupancy()



def adaBoost_final_optimization_USPS():
    path = 'USPS_digits' 
    classifier_name = "AdaBoost"
    train_x, train_y, test_x = get_data(path)
    
    classifier = AdaBoostClassifier()


    tuned_parameters = [{  'classifier__base_estimator': [  RandomForestClassifier()],
                         'classifier__base_estimator__bootstrap': [True], 
                         'classifier__base_estimator__class_weight': [None], 
                         'classifier__base_estimator__criterion': ['gini', 'entropy'],
                         'classifier__base_estimator__max_depth': range(3,200,20), 
                         'classifier__base_estimator__max_features': ['auto'], 
                         #'classifier__base_estimator__max_leaf_nodes': [1],
                         'classifier__base_estimator__min_samples_leaf': [1,2,], 
                         'classifier__base_estimator__min_samples_split': [1,2,3,],
                         'classifier__base_estimator__warm_start':[False],
                         'classifier__n_estimators': range(15,21,5),
                         'classifier__learning_rate': range(1,3,1),#range(0.5,1.0,0.1),
                         'classifier__algorithm' : ['SAMME', 'SAMME.R']

        }            
    ]


    pipe = Pipeline(steps=[('classifier', classifier)])
    clf = GridSearchCV(pipe, param_grid=tuned_parameters, scoring='accuracy', cv = 10)
    
    # For fitting again, these need to be uncommented but takes a lot of time
    #clf.fit(train_x, train_y)
    #print clf.best_score_
    #print  clf.best_estimator_
    
    #0.950564971751
    clf_best_params = AdaBoostClassifier(algorithm='SAMME',
          base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=123, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            warm_start=False),
          learning_rate=1, n_estimators=15, random_state=None)
    
   
    clf_best_params.fit(train_x, train_y)

    scores = cross_val_score(clf_best_params, train_x, train_y, cv=10)

    final_prediction = clf_best_params.predict(test_x)

    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    kaggleize(final_prediction,"../Predictions/"+path+"/Occupancy_detection_AdaBoost_Final_Optimization_Best.csv")



#adaBoost_final_optimization_USPS()



