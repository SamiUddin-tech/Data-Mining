# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:19:20 2021

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

clust=pd.read_excel(r"C:\Users\User\Desktop\clust.xlsx")

# Specifying Columns name

clust=pd.DataFrame(clust)
columns=["X", "Y", "Z"]
clust.columns=columns
#print(clust)
X=clust["X"]
Y=clust["Y"]
Z=clust["Z"]
X1=np.array(list(zip(X,Y,Z)))

# 2D plotting of Dataset

plt.figure(1)
plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# 3D plotting of Dataset

fig2 = plt.figure(2)
plot2 = Axes3D(fig2)
plot2.scatter(X, Y, Z)
plot2.set_xlabel("X")
plot2.set_ylabel("Y")
plot2.set_zlabel("Z")

# Determining no. of clusters Elbow Method

plt.figure(3)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12))
visualizer.fit(clust)        # Fit the data to the visualizer
visualizer.show()

# Using K-means command of sklear.cluster

# Number of clusters
kmeans = KMeans(n_clusters=4)
# Fitting the input data
kmeans = kmeans.fit(X1[:,:3])
# Getting the cluster labels
labels = kmeans.predict(X1[:,:3])
# Centroid values
centroids = kmeans.cluster_centers_
kmeans.labels_
print("Centeroids : ",centroids)
#print(labels)
print(kmeans)
print("Labels: ",kmeans.labels_)

# 3D Plotting of clusters and cluster centres for each cluster 

fig3 = plt.figure(4)
plot3 = Axes3D(fig3)
plot3.scatter(X1[:, 0], X1[:, 1], X1[:,2],c=kmeans.labels_,cmap='rainbow')
plot3.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='k', s=500)
plot3.set_xlabel("X")
plot3.set_ylabel("Y")
plot3.set_zlabel("Z")
plt.show()







