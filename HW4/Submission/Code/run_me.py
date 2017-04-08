import numpy as np
import matplotlib.pyplot as plt
import cluster_utils
from cluster_class import cluster_class
from cluster_class_bonus import cluster_class_bonus
from sklearn.cluster import KMeans
import plot

#Load train and test data
train = np.load("../../Data/ECG/train.npy")
test = np.load("../../Data/ECG/test.npy")

#Create train and test arrays
Xtr = train[:,0:-1]
Xte = test[:,0:-1]
Ytr = np.array(map(int, train[:,-1]))
Yte = np.array( map(int, test[:,-1]))
#print "x"
#Add your code below



##########################################################################
# Question 1
##########################################################################

# K = 40
# scores = [0.0] * K
# for k in range(1,K+1):
#
#     # KMeans with no. of clusters = k
#     cluster = KMeans(n_clusters=k, random_state=10)
#     cluster.fit(Xtr)
#     scores[k - 1] = cluster_utils.cluster_quality(Xtr, cluster.labels_, k)
#
# plot.line_graph(range(1,K+1), scores, "Cluster Score VS K", "OnePointFour", "Score", "K")



##########################################################################
# Question 2
##########################################################################

# K = 6
#
# cluster = KMeans(n_clusters=K, random_state=10)
# cluster.fit(Xtr)
#
# proportions = cluster_utils.cluster_proportions(cluster.labels_, K)
# plot.bar_graph(proportions, "Cluster Proporstions", "TwoPointTwo")
#
# # Compute cluster means
# means = cluster_utils.cluster_means(Xtr, cluster.labels_, K)
# # Show cluster means
# cluster_utils.show_means(means, proportions).savefig("../Figures/TwoPointFourBarChart.pdf")


##########################################################################
# Question 3
##########################################################################
# K = 40
# scores = [0.0]*K
# for k in range(1,K +1):
#     cluster = cluster_class(k)
#     cluster.fit(Xtr,Ytr)
#     scores[k-1] = cluster.score(Xte,Yte)
#
# plot.line_graph(range(1,K +1), scores, "Prediction Error vs No. of Clusters", "ThreePointFive", "Error", "No. of Clusters")
#




##########################################################################
# Question 4
##########################################################################

K = 40
scores = [0.0]*K
for k in range(1,K +1):
    cluster = cluster_class_bonus(k)
    cluster.fit(Xtr,Ytr)
    scores[k-1] = cluster.score(Xte,Yte)

plot.line_graph(range(1,K +1), scores, "Prediction Error vs No. of Clusters", "ThreePointSeven", "Error", "No. of Clusters")




