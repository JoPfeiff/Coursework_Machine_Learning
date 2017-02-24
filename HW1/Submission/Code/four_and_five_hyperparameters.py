

from sklearn import neighbors
from sklearn import tree
import numpy as np
import random
import itertools as it
import warnings
import csv
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC

warnings.filterwarnings('ignore')



def get_data(path):
    
    #path = 'Data/Email_spam/'
    train = np.load("../../Data/" + path + '/train.npy')
    test = np.load("../../Data/" + path + '/test_distribute.npy')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    test_x = test[:, 1:]
    
    return train_x, train_y, test_x


# Class that splits the data set into the defined amount of buckets
# and does brute force search with all defined parameters
class GridSearchCV_new():    
    
    def __init__(self, clf, parameters, cv):
        
        # the defined classifier
        self.clf = clf
        
        # the parameters which should be compared
        self.parameters = parameters
        
        # The amounts of buckets
        self.cv = cv   
        
        # The best average accuracy a set parameter has found
        self.accuracy     = None
        
        # All accuracy scores of buckets of the best parameters
        self.best_scores     = None
        
        # The best parameters
        self.best_parameter = None
        
        # All scores of all parameters
        self.all_scores = {}
        self.all_mean_scores = {}
    
    # This method should be called to fit the classifier with training sets
    # train_x are all the features
    # train_y are all the lables
    def fit(self, train_x, train_y):
        
        # make it into a numpy array 
        np_train_x = np.array(train_x)
        np_train_y = np.array(train_y)   
        
        # split the data into the with CV defined buckets
        split_data = self.__split__(np_train_x,np_train_y)
        
        # loop through the buckets and parameters to find the best parameter settings
        self.__generate_model_loop__(split_data)
        
    
    # This method splits the data into the defined amount of buckets
    def __split__(self, train_x, train_y):
        # first add the lables to the training set so that the shuffeling is easier
        data = np.c_[train_x , train_y.T]
        #shuffle the data set
        random.shuffle(data) 
        # split the dataset into the with cv defined amount of buckets
        split_data = np.array_split(data,self.cv,0)
        return split_data
    
    # This method generates all possible parameter settings
    def generate_parameters(self, all_params = None):
        if (all_params is None):
            all_params = self.parameters
        # first generate all parametersettings
        iterables = [all_params[param] for param in all_params]
        # then also generate all parameter definitions
        keys = [param for param in all_params]
        # then generate a dictionary with all combinations
        return [{keys[i]:tup[i]for i in range(0,len(keys))} for tup in it.product(*iterables)]
    
    # This method loops through all buckets and parameter settings
    def __generate_model_loop__(self, split_data):
        
        parameters = self.generate_parameters()
        buckets = self.__generate_buckets__(split_data)

        
        # best found parameter
        best_parameter = None
        best_score = 0.0
        best_scores = [0.0] * self.cv
        
        
        # loop through all parameters
        
        for parameter in parameters:
            # array for all the current scores per bucket
            current_scores = [0.0] * self.cv
            
            # generate the model and predict for all buckets with current parameter
            for i in range(0,self.cv):
                #save current scores and current_parameters
                current_scores[i], current_parameters = self.__generate_model__(parameter,buckets[i])
            # calculate the mean of accuracy for bucket with set parameters
            mean = np.mean(current_scores)
            #print mean
            #print parameter
            #print "\n"
            self.all_mean_scores[str(parameter)] = mean
            self.all_scores[str(parameter)] = current_scores
            # if scores are the best until now, save everything
            if (self.accuracy < mean):
                #print str(mean) + " accuracy = " + str(mean)
                #print current_parameters
                
                self.accuracy = mean
                self.best_scores = current_scores
                #print type(self.best_parameter)
                self.best_parameter = parameter#str(current_parameters)
                
            
        
        #save overall best scores to class variables
        #self.accuracy       = best_score
        #self.best_scores    = best_scores
        #self.best_parameter = best_parameter
    
    
    # generate model which returns accuracy of bucket
    def __generate_model__(self, parameter, bucket):
        # define classifier
        classifier = self.clf
        # set the parameters for current classifier
        classifier.set_params(**parameter)
        #fit the classifier with the split up buckets training features, and training lables (very end)
        classifier.fit(bucket[0][:, :-1], bucket[0][:, -1:])
        # predict the test data with the generated model on the test data bucket[1]
        prediction = classifier.predict(bucket[1][:, :-1])
        correct = 0
        # calculate the accuracy of the prediction and return ist
        for i in range(0,len(prediction)):
            if (prediction[i] == bucket[1][:, -1:][i]):
                correct += 1
        #print correct/ (len(prediction) * 1.0)
        return correct/(len(prediction)*1.0), classifier.get_params
    
    #Thist method generates the buckets for training and testing from the split data
    def __generate_buckets__(self, split_data):
        # define the size of the bucket
        bucket = [None] * self.cv
        # loop through the amount of buckets that will be created
        for i in range (0,self.cv):
            # define the lower bucket part - one training and test set
            bucket_part = [None]*2
            part = 0
            # loop through the split_data to generate the train and test sets
            for a in split_data: 
                # when i and part are the same, this block will become the test set
                # that way every block will become a test set exactly once
                if part == i:
                    bucket_part[1] = a
                # if there is nothing in the training set yet, we will fill it
                elif(bucket_part[0] is None):
                    bucket_part[0] = a
                # else we will add to it
                else:
                    bucket_part[0] = np.vstack([bucket_part[0], a])
                part += 1
            # add the current train and test set to the overall bucket
            bucket[i] = bucket_part
        return bucket
          
    def get_accuracy(self):
        if(self.accuracy is None):
            print ("Fit the Classifier first")
        else:
            return self.accuracy
    
    def get_best_scores(self):
        if(self.best_scores is None):
            print ("Fit the Classifier first")
        else:
            return self.best_scores
    
    def get_best_parameter(self):
        if(self.best_parameter is None):
            print ("Fit the Classifier first")
        else:
            return self.best_parameter
        
    def get_all_mean_scores(self):
        if(self.all_mean_scores is None):
            print ("Fit the Classifier first")
        else:
            return self.all_mean_scores
        
    def get_all_scores(self):
        if(self.all_scores is None):
            print ("Fit the Classifier first")
        else:
            return self.all_scores
        
    def get_all_means_and_error_scores(self):
        all_error_scores = [0.0] * len(self.all_mean_scores)
        all_m_scores = [0.0] * len(self.all_mean_scores)
        if(self.all_mean_scores is None):
            print ("Fit the Classifier first")
        else:
            i=0
            for  key, value in  self.all_mean_scores.iteritems():
                #print self.all_mean_scores[1][1]
                all_error_scores[i] = 1.0 - value
                all_m_scores[i] = value
                i+=1
            return [all_m_scores, all_error_scores]
    
    



# In[54]:

def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions)).astype(int)
    writer = csv.writer(open(file, 'w'))
    writer.writerow(['# id','Prediction'])
    writer.writerows(kaggle_predictions)


# In[75]:

#general optimization method for faster optimization of different classifiers
def optimize_parameters(path, classifier, features, cv, classifier_name):
    # get data
    train_x, train_y, test_x = get_data(path)
    
    #define gridsearch object
    gsCV = GridSearchCV_new(classifier,features,cv)
    
    gsCV.fit(train_x, train_y)
    print "\n\n"+path + ": " + classifier_name + " Optimized:"
    print gsCV.get_accuracy()
    print gsCV.get_best_scores()
    print gsCV.get_best_parameter()
    gsCV.get_all_mean_scores()
    gsCV.get_all_scores()
    
    # when optimized train again with optimal params so that we dont loose data for training
    classifier.set_params(**gsCV.get_best_parameter())
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(test_x)
    
    # save to file -> prediction and graph
    file = "../Predictions/" + path +"/" + classifier_name+"_best.csv"
    kaggleize(prediction,file)
    title = path + "_" + classifier_name + "_accuracy_and_error"
    print title
    plot_line_graph(gsCV.get_all_means_and_error_scores(), ['Error Score', 'Accuracy Score'], title)
    


def optimize_knn_email():
    
    path = 'Email_spam'  
    classifier_name = "KNN"
    features = {'n_neighbors': range(1,20,2),
                 'metric': ['minkowski','manhattan','euclidean','chebyshev']}
    classifier = neighbors.KNeighborsClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #BEST
    #Email_spam: KNN Optimized:
    #0.935386436768
    #[0.9945054945054945, 0.989010989010989, 0.967032967032967, 0.9613259668508287, 0.9060773480662984, 0.9005524861878453, 0.9226519337016574, 0.9392265193370166, 0.9060773480662984, 0.8674033149171271]
    #{'n_neighbors': 1, 'metric': 'manhattan'}
    
    
def acc_default_knn_email():
    
    path = 'Email_spam'  
    classifier_name = "Default_KNN"
    features = {'n_neighbors': [5]}
    classifier = neighbors.KNeighborsClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #BEST
    #Email_spam: KNN Optimized:
    #0.935386436768
    #[0.9945054945054945, 0.989010989010989, 0.967032967032967, 0.9613259668508287, 0.9060773480662984, 0.9005524861878453, 0.9226519337016574, 0.9392265193370166, 0.9060773480662984, 0.8674033149171271]
    #{'n_neighbors': 1, 'metric': 'manhattan'}
    
    
def optimize_knn_occupancy():
    
    path = 'Occupancy_detection'
    classifier_name = "KNN"
    features = {'n_neighbors': range(1,30,2),
                 'metric': ['minkowski','manhattan','euclidean','chebyshev']}
    classifier = neighbors.KNeighborsClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #BEST:
    #Occupancy_detection: KNN Optimized:
    #0.996631578947
    #[1.0, 1.0, 0.9978947368421053, 0.9978947368421053, 0.9894736842105263, 0.9936842105263158, 1.0, 0.991578947368421, 0.9957894736842106, 1.0]
    #{'n_neighbors': 1, 'metric': 'minkowski'}
    


    
def optimize_knn_USPS():
    
    path = 'USPS_digits'
    classifier_name = "KNN"
    features = {'n_neighbors': range(1,35,2),
                 'metric': ['minkowski','manhattan','euclidean','chebyshev']}
    classifier = neighbors.KNeighborsClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #BEST
    #USPS_digits: KNN Optimized:
    #0.984463276836
    #[0.9915254237288136, 0.9971751412429378, 0.9887005649717514, 0.9943502824858758, 0.9887005649717514, 0.9830508474576272, 0.980225988700565, 0.9661016949152542, 0.9745762711864406, 0.980225988700565]
    #{'n_neighbors': 1, 'metric': 'minkowski'}

    

def optimize_decision_tree_Email():
    
    path = 'Email_spam' 
    classifier_name = "Decision_Tree"
    features = {'criterion': ['entropy','gini'],
                'max_depth': range(5,20,3),
                'min_samples_split': range(1,10,2)}
    classifier = tree.DecisionTreeClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #BEST:
    #Email_spam: Decision_Tree Optimized:
    #0.963038066905
    #[0.9615384615384616, 0.967032967032967, 0.9725274725274725, 0.9779005524861878, 0.9723756906077348, 0.9779005524861878, 0.9558011049723757, 0.9502762430939227, 0.9613259668508287, 0.9337016574585635]
    #{'min_samples_split': 1, 'criterion': 'gini', 'max_depth': 14}

def optimize_decision_tree_occupancy():
    
    path = 'Occupancy_detection' 
    classifier_name = "Decision_Tree"
    features = {'criterion': ['entropy','gini'],
                'max_depth': range(5,50,3),
                'min_samples_split': range(1,10,2)}
    classifier = tree.DecisionTreeClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #BEST
    #Occupancy_detection: Decision_Tree Optimized:
    #0.997263157895
    #[0.9978947368421053, 1.0, 0.9978947368421053, 0.9978947368421053, 0.9957894736842106, 0.9978947368421053, 0.9957894736842106, 1.0, 0.9957894736842106, 0.9936842105263158]
    #{'min_samples_split': 1, 'criterion': 'entropy', 'max_depth': 17}
    

def optimize_decision_tree_USPS():
    
    path = 'USPS_digits'
    classifier_name = "Decision_Tree"
    features = {'criterion': ['entropy','gini'],
                'max_depth': range(5,20,3),
                'min_samples_split': range(1,6,2)}
    classifier = tree.DecisionTreeClassifier()
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    #USPS_digits: Decision_Tree Optimized:
    #0.946892655367
    #[0.980225988700565, 0.980225988700565, 0.980225988700565, 0.9519774011299436, 0.963276836158192, 0.9519774011299436, 0.940677966101695, 0.9152542372881356, 0.9265536723163842, 0.8785310734463276]
    #{'min_samples_split': 1, 'criterion': 'gini', 'max_depth': 14}

    
def optimize_AdaBoost_Email():
    
    path = 'Email_spam' 
    classifier_name = "AdaBoost"   
    classifier = AdaBoostClassifier()
    train_x, train_y, test_x = get_data(path)
    features = {'base_estimator': [tree.DecisionTreeClassifier()],
                'n_estimators': range(1,55,5),
                'learning_rate': range(1,10,2),
                'algorithm' : ['SAMME', 'SAMME.R']
                 }    
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)    
    
    #BEST:
    #Email_spam: AdaBoost Optimized:
    #0.955282010807
    #[0.9725274725274725, 0.978021978021978, 0.989010989010989, 0.9834254143646409, 0.9668508287292817, 0.9668508287292817, 0.9226519337016574, 0.9337016574585635, 0.9447513812154696, 0.8950276243093923]
    #{'n_estimators': 36, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            #max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            #min_samples_split=2, min_weight_fraction_leaf=0.0,
            #presort=False, random_state=None, splitter='best'), 'learning_rate': 3, 'algorithm': 'SAMME'}
    
    


def optimize_AdaBoost_occupancy():
    
    path = 'Occupancy_detection' 
    classifier_name = "AdaBoost"
    classifier = AdaBoostClassifier()
    train_x, train_y, test_x = get_data(path)
    features = {'base_estimator': [tree.DecisionTreeClassifier()],
                'n_estimators': range(1,55,5),
                'learning_rate': range(1,10,2),
                'algorithm' : ['SAMME', 'SAMME.R']
                 }    
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    
    #BEST
    #Occupancy_detection: AdaBoost Optimized:
    #0.996842105263
    #[0.9957894736842106, 1.0, 0.9957894736842106, 0.9978947368421053, 0.991578947368421, 0.9957894736842106, 1.0, 0.9978947368421053, 0.9936842105263158, 1.0]
    #{'n_estimators': 1, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            #max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            #min_samples_split=2, min_weight_fraction_leaf=0.0,
            #presort=False, random_state=None, splitter='best'), 'learning_rate': 3, 'algorithm': 'SAMME.R'}

    
    

def optimize_AdaBoost_USPS():
        
    path = 'USPS_digits'
    classifier_name = "AdaBoost"
    classifier = AdaBoostClassifier()
    train_x, train_y, test_x = get_data(path)
    features = {'base_estimator': [tree.DecisionTreeClassifier()],
                'n_estimators': range(1,20,5),
                'learning_rate': range(1,5,2),
                'algorithm' : ['SAMME', 'SAMME.R']
                 }    
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    
    #BEST:
    #USPS_digits: AdaBoost Optimized:
    #0.94406779661
    #[0.9774011299435028, 0.9689265536723164, 0.9661016949152542, 0.9519774011299436, 0.9689265536723164, 0.9548022598870056, 0.9378531073446328, 0.9124293785310734, 0.9124293785310734, 0.8898305084745762]
    #{'n_estimators': 1, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            #max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            #min_samples_split=2, min_weight_fraction_leaf=0.0,
            #presort=False, random_state=None, splitter='best'), 'learning_rate': 5, 'algorithm': 'SAMME'}
    

    
def optimize_SVM_Email():
    
    path = 'Email_spam' 
    classifier_name = "SVM"   
    classifier = SVC()
    train_x, train_y, test_x = get_data(path)
    features = { 'C':[0.8,0.9,1.0], 
                'cache_size':[190,200], 
                'class_weight':[None], 
                'coef0':[0.0],
                'decision_function_shape':[None], 
                'degree':[2,3,4], 
                'gamma':['auto'], 
                'kernel':['rbf','linear', 'poly'],
                'max_iter':[-1], 
                'probability':[False], 
                'random_state':[None], 
                'shrinking':[True],
                'tol':[0.001], 
                'verbose':[False]}   
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)    
    
    #BEST:



def optimize_SVM_occupancy():
    
    path = 'Occupancy_detection' 
    classifier_name = "SVM"
    classifier = SVC()
    train_x, train_y, test_x = get_data(path)
    features = { 'C':[0.8,0.9,1.0], 
                'cache_size':[190,200], 
                'class_weight':[None], 
                'coef0':[0.0],
                'decision_function_shape':[None], 
                'degree':[2,3,4], 
                'gamma':['auto'], 
                'kernel':['rbf','linear', 'poly'],
                'max_iter':[-1], 
                'probability':[False], 
                'random_state':[None], 
                'shrinking':[True],
                'tol':[0.001], 
                'verbose':[False]}    
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    
    #BEST

    

def optimize_SVM_USPS():
        
    path = 'USPS_digits'
    classifier_name = "SVM"
    classifier = SVC()
    train_x, train_y, test_x = get_data(path)
    features =  { 'C':[0.8,0.9,1.0], 
                'cache_size':[190,200], 
                'class_weight':[None], 
                'coef0':[0.0],
                'decision_function_shape':[None], 
                'degree':[2,3,4], 
                'gamma':['auto'], 
                'kernel':['rbf','linear', 'poly'],
                'max_iter':[-1], 
                'probability':[False], 
                'random_state':[None], 
                'shrinking':[True],
                'tol':[0.001], 
                'verbose':[False]}  
    cv = 10    
    optimize_parameters(path, classifier, features, cv,classifier_name)
    
    #BEST:



# plot the line graph
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

    plt.legend(labels, loc = 'center')

    plt.savefig("../Figures/"+title_img+".pdf")



   



# In[ ]:

#optimize_knn_email()
#acc_default_knn_email()
#optimize_knn_occupancy()
#optimize_knn_USPS()
#optimize_decision_tree_Email()
#optimize_decision_tree_occupancy()
#optimize_decision_tree_USPS()
#optimize_AdaBoost_Email()
#optimize_AdaBoost_occupancy()
#optimize_AdaBoost_USPS()
#optimize_SVM_Email()
#optimize_SVM_occupancy()
#optimize_SVM_USPS()






