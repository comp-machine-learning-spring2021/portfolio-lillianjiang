# import block
import pandas as pd
import numpy as np
from scipy.spatial import distance    
from sklearn.cluster import KMeans 

# looping_kmeans
def looping_kmeans(arr,k_cluster):
    sums=[]
    
    for k in k_cluster:
        km_alg = KMeans(n_clusters=k, init="random",random_state = 1, max_iter = 300)
        fit3 = km_alg.fit(arr)
        labels = fit3.labels_
        centers = fit3.cluster_centers_

        within_cluster_sumsqs = 0
        n = 0
        for c in centers:
            # Extract the cluster's center and associated points:
            cluster_center = c.reshape(1,-1)
            cluster_points = arr[labels==n]
            n = n+1

            # Compute the following for each cluster:
            cluster_spread = distance.cdist(cluster_points, cluster_center, 'euclidean')
            cluster_total = np.sum(cluster_spread)

            # Add this cluster's within sum of squares to within_cluster_sumsqs
            within_cluster_sumsqs = within_cluster_sumsqs + cluster_total
        sums.append(within_cluster_sumsqs)
    return sums

# my from scratch implementation of KMeans
def kMeans(data_np, k, max_iteration):
    # Randomly select some datapoints to be the center
    idx = np.random.randint(data_np.shape[0], size=k)
    centers_np = data_np[idx,:]
    
    iteration = 0
    while iteration < max_iteration:
        # Calulate the distance from points to centers
        dists = distance.cdist(data_np, centers_np, 'euclidean')
        labels = np.argmin(dists, axis=1)

        # Update the centers
        for i in range(k):
            cluster = data_np[labels[:] == i]
            centers_np[i,0] = cluster[:,0].mean()
            centers_np[i,1] = cluster[:,1].mean()
        iteration += 1
    
    return centers_np,labels