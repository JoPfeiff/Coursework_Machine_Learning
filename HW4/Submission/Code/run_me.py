import numpy as np
import matplotlib.pyplot as plt
import cluster_utils
import cluster_class

#Load train and test data
train = np.load("../../Data/ECG/train.npy")
test = np.load("../../Data/ECG/test.npy")

#Create train and test arrays
Xtr = train[:,0:-1]
Xte = test[:,0:-1]
Ytr = np.array(map(int, train[:,-1]))
Yte = np.array( map(int, test[:,-1]))
print "x"
#Add your code below