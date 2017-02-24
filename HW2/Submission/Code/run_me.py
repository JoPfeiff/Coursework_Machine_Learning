# run_me.py module

import kaggle

# Assuming you are running run_me.py from the Submission/Code directory, otherwise the path variable will be different for you
import numpy as np

# Load the Air Foil data
path = '../../Data/AirFoil/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test_private.npy')
train_x = train[:, 0:train.shape[1]-1]
train_y = train[:, -1]
test_x = test[:, 0:test.shape[1]-1]

# Real test targets/outputs
test_y = test[:, -1]
print "Air Foil:", train_x.shape, train_y.shape, test_x.shape, test_y.shape 


# Load the Blog feedback data
path = '../../Data/BlogFeedback/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test_distribute.npy')
train_x = train[:, 0:train.shape[1]-1]
train_y = train[:, -1]
test_x = test[:, 0:test.shape[1]-1]
# this is dummy vector of predictions
test_y = test[:, -1]
print "Blog Feedback:", train_x.shape, train_y.shape, test_x.shape, test_y.shape 

#Save prediction file in Kaggle format for scoring on Kaggle
predictions = np.zeros(test_y.shape)
kaggle.kaggleize(predictions, "../Predictions/BlogFeedback/test.csv")





