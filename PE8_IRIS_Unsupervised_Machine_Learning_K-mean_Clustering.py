# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 01:19:05 2021

@author: User
"""
# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Loading iris datset

fisher_iris=pd.read_csv(r"C:\Users\User\Downloads\iris.data")

# Specifying Columns name

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
fisher_iris.columns = attributes
print("Head of Fisher Dataset : \n",fisher_iris.head())
petal_length=fisher_iris["petal_length"]
petal_width=fisher_iris["petal_width"]
sepal_length=fisher_iris["sepal_length"]
sepal_width=fisher_iris["sepal_width"]
X=np.array(list(zip(petal_length,petal_width, sepal_width,sepal_length)))

# 2D plotting of Petal Length and Petal Width 

plt.figure(1)
plt.scatter(petal_length, petal_width)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.grid(True)
plt.show()

# Determining no. of clusters Elbow Method 

plt.figure(2)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12))
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 

# 3D plotting of PetalLength, Petal Width and Sepal width 

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:,2])
ax.set_xlabel("petal Length")
ax.set_ylabel("petal width")
ax.set_zlabel("sepal width")

# Using K-means command of sklear.cluster

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X[:,:4])
# Getting the cluster labels
labels = kmeans.predict(X[:,:4])
# Centroid values
centroids = kmeans.cluster_centers_
kmeans.labels_
print("Centroids : ",centroids)
print("Cluster Labels : ",labels)
print(kmeans)
#print(kmeans.labels_)

# 3D Plotting of clusters and cluster centres for each cluster 

fig2 = plt.figure(4)
bx = Axes3D(fig2)
bx.scatter(X[:, 0], X[:, 1], X[:,2],c=kmeans.labels_,cmap='rainbow')
bx.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='k', s=500)
bx.set_xlabel("petal Length")
bx.set_ylabel("petal width")
bx.set_zlabel("sepal width")
plt.show()

