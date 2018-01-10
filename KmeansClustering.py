#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:22:40 2018

@author: virajdeshwal
"""

print('Lets begin with the Kmeans Clustering.\n')
#intake = input('Press any key to continue....\n\n')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv('Mall_Customers.csv')

X = file.iloc[:,[3,4]].values


'''to find the optimal clusters use Elbow method... remove these comments and use the beolw code to check the elbow graph
# Now lets use the Elbow method to define the optioal number of clusters



#metric for clusters
wcss = []
from sklearn.cluster import KMeans
#for loop to check the clusters from 1 to 10
for i in range(1,11):
    #intialization of the model
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
    #fitting the kmeans to the independent variables
    #Now lets calculate the centroid of the cluster
    wcss.append(kmeans.inertia_)
    

plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')
plt.xlabel('Numbers of Clusters')
plt.ylabel('WCSS')
plt.show()'''


'''Now as we got the idea from the elbow graph about the optimal no. of clusters.
    we will take the 5 clusters for our dataset.'''
    
#applying k-means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_means =kmeans.fit_predict(X)
plt.scatter(X[y_means==0,0], X[y_means==0,1], s=100, c='red', label = 'Careful pals')
plt.scatter(X[y_means==1,0], X[y_means==1,1], s=100, c='blue', label = 'average')
plt.scatter(X[y_means==2,0], X[y_means==2,1], s=100, c='green', label = 'Targets')
plt.scatter(X[y_means==3,0], X[y_means==3,1], s=100, c='magenta', label = 'Freak')
plt.scatter(X[y_means==4,0], X[y_means==4,1], s=100, c='cyan', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label = 'Centroids')

plt.title('clusters of client')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print('\nDone ;)')
