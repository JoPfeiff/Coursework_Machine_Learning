import numpy as np
import random
import itertools as it
import warnings
warnings.filterwarnings('ignore')



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
        self.accuracy = None

        # All accuracy scores of buckets of the best parameters
        self.best_scores = None

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
        split_data = self.__split__(np_train_x, np_train_y)

        # loop through the buckets and parameters to find the best parameter settings
        self.__generate_model_loop__(split_data)

    # This method splits the data into the defined amount of buckets
    def __split__(self, train_x, train_y):
        # first add the lables to the training set so that the shuffeling is easier
        data = np.c_[train_x, train_y.T]
        # shuffle the data set
        random.shuffle(data)
        # split the dataset into the with cv defined amount of buckets
        split_data = np.array_split(data, self.cv, 0)
        return split_data

    # This method generates all possible parameter settings
    def generate_parameters(self, all_params=None):
        if (all_params is None):
            all_params = self.parameters
        # first generate all parametersettings
        iterables = [all_params[param] for param in all_params]
        # then also generate all parameter definitions
        keys = [param for param in all_params]
        # then generate a dictionary with all combinations
        return [{keys[i]: tup[i] for i in range(0, len(keys))} for tup in it.product(*iterables)]

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
            for i in range(0, self.cv):
                # save current scores and current_parameters
                current_scores[i], current_parameters = self.__generate_model__(parameter, buckets[i])
            # calculate the mean of accuracy for bucket with set parameters
            mean = np.mean(current_scores)
            # print mean
            # print parameter
            # print "\n"
            self.all_mean_scores[str(parameter)] = mean
            self.all_scores[str(parameter)] = current_scores
            # if scores are the best until now, save everything
            if (self.accuracy < mean):
                # print str(mean) + " accuracy = " + str(mean)
                # print current_parameters

                self.accuracy = mean
                self.best_scores = current_scores
                # print type(self.best_parameter)
                self.best_parameter = parameter  # str(current_parameters)



                # save overall best scores to class variables
                # self.accuracy       = best_score
                # self.best_scores    = best_scores
                # self.best_parameter = best_parameter

    # generate model which returns accuracy of bucket
    def __generate_model__(self, parameter, bucket):
        # define classifier
        classifier = self.clf
        # set the parameters for current classifier
        classifier.set_params(**parameter)
        # fit the classifier with the split up buckets training features, and training lables (very end)
        classifier.fit(bucket[0][:, :-1], bucket[0][:, -1:])
        # predict the test data with the generated model on the test data bucket[1]
        prediction = classifier.predict(bucket[1][:, :-1])
        correct = 0
        # calculate the accuracy of the prediction and return ist
        for i in range(0, len(prediction)):
            if (prediction[i] == bucket[1][:, -1:][i]):
                correct += 1
        # print correct/ (len(prediction) * 1.0)
        return correct / (len(prediction) * 1.0), classifier.get_params

    # Thist method generates the buckets for training and testing from the split data
    def __generate_buckets__(self, split_data):
        # define the size of the bucket
        bucket = [None] * self.cv
        # loop through the amount of buckets that will be created
        for i in range(0, self.cv):
            # define the lower bucket part - one training and test set
            bucket_part = [None] * 2
            part = 0
            # loop through the split_data to generate the train and test sets
            for a in split_data:
                # when i and part are the same, this block will become the test set
                # that way every block will become a test set exactly once
                if part == i:
                    bucket_part[1] = a
                # if there is nothing in the training set yet, we will fill it
                elif (bucket_part[0] is None):
                    bucket_part[0] = a
                # else we will add to it
                else:
                    bucket_part[0] = np.vstack([bucket_part[0], a])
                part += 1
            # add the current train and test set to the overall bucket
            bucket[i] = bucket_part
        return bucket

    def get_accuracy(self):
        if (self.accuracy is None):
            print ("Fit the Classifier first")
        else:
            return self.accuracy

    def get_best_scores(self):
        if (self.best_scores is None):
            print ("Fit the Classifier first")
        else:
            return self.best_scores

    def get_best_parameter(self):
        if (self.best_parameter is None):
            print ("Fit the Classifier first")
        else:
            return self.best_parameter

    def get_all_mean_scores(self):
        if (self.all_mean_scores is None):
            print ("Fit the Classifier first")
        else:
            return self.all_mean_scores

    def get_all_scores(self):
        if (self.all_scores is None):
            print ("Fit the Classifier first")
        else:
            return self.all_scores

    def get_all_means_and_error_scores(self):
        all_error_scores = [0.0] * len(self.all_mean_scores)
        all_m_scores = [0.0] * len(self.all_mean_scores)
        if (self.all_mean_scores is None):
            print ("Fit the Classifier first")
        else:
            i = 0
            for key, value in self.all_mean_scores.iteritems():
                # print self.all_mean_scores[1][1]
                all_error_scores[i] = 1.0 - value
                all_m_scores[i] = value
                i += 1
            return [all_m_scores, all_error_scores]


