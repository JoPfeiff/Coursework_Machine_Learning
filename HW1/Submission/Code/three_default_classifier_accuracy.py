

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.svm import SVC
import warnings
import csv
import time
warnings.filterwarnings('ignore')



#Load all the data into np arrays
def get_data(path):
    
    #path = 'Data/Email_spam/'
    train = np.load("../../Data/" + path + '/train.npy')
    test = np.load("../../Data/" + path + '/test_distribute.npy')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    test_x = test[:, 1:]
    
    return train_x, train_y, test_x



#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions)).astype(int)
    writer = csv.writer(open(file, 'w'))
    writer.writerow(['# id','Prediction'])
    writer.writerows(kaggle_predictions)

# this has all the accuracy from kaggle saved
def get_hardcoded_accuracy(classifier,data):
    accuracy = {'Email_spamKNN': 0.77704,
                'Email_spamDecision Tree': 0.88300,
                'Email_spamAdaBoost': 0.94040,
                'Email_spamSVC': 0.80353,
                'USPS_digitsKNN': 0.98989,
                'USPS_digitsDecision Tree': 0.98653,
                'USPS_digitsAdaBoost': 0.99074,
                'USPS_digitsSVC': 0.79453,
                'Occupancy_detectionKNN': 0.96836,
                'Occupancy_detectionDecision Tree': 0.82938,
                'Occupancy_detectionSVC': 0.95876,
                'Occupancy_detectionAdaBoost': 0.30056}
    return accuracy[classifier+data]
    
def question_3():
    
    # paths to the data
    paths = ['Email_spam', 'USPS_digits', 'Occupancy_detection']
    
    #the three classifiers I am using
    classifiers = [(neighbors.KNeighborsClassifier(), "KNN"), (tree.DecisionTreeClassifier(),"Decision Tree"), (AdaBoostClassifier(),"AdaBoost"), (SVC(),"SVC")]

    # for all paths run
    for path in paths:
        
        # get the data from all the paths
        train_x, train_y, test_x = get_data(path)
        
        # run the classification for all data for all classifiers
        for classifier_tup in classifiers:
            
            #get the classifier
            classifier = classifier_tup[0]
            
            # get the classifier name
            classifier_name = classifier_tup[1]
            
            print path + " : " + classifier_name
            
            # time for prediction and fit the classifier
            start = time.time()            
            classifier.fit(train_x, train_y)
            end = time.time()    
            modeling_time = end - start
            print "Time for modeling: " + str(modeling_time)
            
            # time for prediction and prediction
            start = time.time()  
            prediction = classifier.predict(test_x)
            end = time.time()     
            predicting_time = end - start
            print "Time for predicting: " + str(predicting_time) 
            
            # print outcome
            print "Accuracy: " + str(get_hardcoded_accuracy(path,classifier_name))+ "\n\n"
            
            # save to file
            file = "../Predictions/" + path +"/" +classifier_name+ "_default_classification.csv"
            kaggleize(prediction,file)
            
# calculation of accuracy
def calculate_accuracy(prediction, label):
    correct = 0
    for i in range(0,len(prediction)):
        print prediction[i] , label[:, -1][i]
        if (prediction[i] == label[:, -1][i]):
            correct += 1
    return correct/(len(prediction)*1.0)

# bar graph generation and saving to pdf for  Training time, prediciton time and accurcay
def bar_graph(values, yLabel, dataset):

    N = len(values)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.30

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects = ax.bar(ind+width, values, width, color='b')
    plt.title(yLabel + " " + dataset)
    ax.set_ylabel(yLabel)
    ax.set_xlabel('Classifiers')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('KNN', 'DT', 'AB') )
    plt.savefig("../Figures/"+yLabel+"_"+dataset +".pdf")

    
def data_set_for_bargraph():
    TrainingTime = [[0.00326800346375, 0.0342168807983, 0.331130981445], [0.0586421489716, 0.934345960617, 6.0385890007], [0.0066351890564, 0.0111861228943, 0.440695047379]]
    PredictionTime =  [[0.0357539653778, 0.00042986869812, 0.0168859958649], [5.93513512611, 0.00205612182617, 0.0902938842773], [0.0314979553223, 0.000391960144043, 0.0295090675354]]
    Accuracy = [[0.77704, 0.883, 0.9404], [0.98989, 0.98653, 0.99074], [0.96836,0.82938, 0.30056]]



    datasets = ['Email Spam Prediction', 'Occupancy Detection', 'USPS Digits Detection']

    for i in range(len(datasets)):
        bar_graph(TrainingTime[i][:], 'Modeling Time',datasets[i])
        bar_graph(PredictionTime[i][:], 'Prediction Time',datasets[i])
        bar_graph(Accuracy[i][:], 'Accuracy',datasets[i])









