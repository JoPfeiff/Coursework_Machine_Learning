import numpy as np
import matplotlib.pyplot as plt
import math
def cluster_quality(X,Z,K):
    '''
    Compute a cluster quality score given a data matrix X (N,D), a vector of 
    cluster indicators Z (N,), and the number of clusters K.
    '''
    score = 0

    for i in range(len(X)):
        for j in range(len(X)):
            if Z[i] == Z[j]:
                score += np.linalg.norm(X[i] - X[j])

    return score / 2
    
def cluster_proportions(Z,K):
    '''
    Compute the cluster proportions p such that p[k] gives the proportion of
    data cases assigned to cluster k in the vector of cluster indicators Z (N,).
    The proportion p[k]=Nk/N where Nk are the number of cases assigned to
    cluster k. Output shape must be (K,)
    '''

    return np.bincount(Z)/ float(len(Z))

        
def cluster_means(X,Z,K):
    '''
    Compute the mean or centroid mu[k] of each cluster given a data matrix X (N,D), 
    a vector of cluster indicators Z (N,), and the number of clusters K.
    mu must be an array of shape (K,D) where mu[k,:] is the average of the data vectors
    (rows) that are assigned to cluster k according to the indicator vector Z.
    If no cases are assigned to cluster k, the corresponding mean value should be zero.
    '''
    D = X.shape[1]
    mu = np.zeros((K,D))

    for i in range(len(X)):
        mu[Z[i]] += X[i]

    bincount = np.bincount(Z)
    for i in range(len(mu)):
        mu[i] = mu[i]/bincount[i]

    return mu
    
def show_means(mu,p):
    '''
    Plot the cluster means contained in mu sorted by the cluster proportions 
    contained in p.
    '''
    K = p.shape[0]
    f = plt.figure(figsize=(8,8))
    for k in range(K):
        plt.subplot(8,5,k+1)
        plt.plot(mu[k,:])
        plt.title("Cluster %d: %.3f"%(k,p[k]),fontsize=5)
        plt.gca().set_xticklabels([])
        plt.gca().set_xticks([25,50,75,100,125,150,175])
        plt.gca().set_yticklabels([])
        plt.gca().set_yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
        plt.ylim(-0.2,1.2)
        plt.grid(True)
    plt.tight_layout()
    return f
        
        
        
    