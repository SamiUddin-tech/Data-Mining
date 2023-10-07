# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:19:21 2021

@author: User
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

# Load the clust dataset

clust=pd.read_excel(r"C:\Users\User\Desktop\clust.xlsx")

attributes = ["X", "Y", "Z"]
clust.columns = attributes
X=round(clust["X"],1)
Y=round(clust["Y"],1)
Z=clust["Z"]

C=np.array(list(zip(X,Y,Z)))
Z = linkage(C[:,:2], 'ward')

labels = C
plt.figure(1,figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(C[:,0],C[:,1], label='True Position')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
for label, x, y in zip(labels, C[:, 0], C[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom', fontsize=8)
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(C)
print(cluster.labels_)
plt.figure(2,figsize=(10, 7))
plt.scatter(C[:,0],C[:,1], c=cluster.labels_, cmap='rainbow')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

fig2 = plt.figure(3)
bx = Axes3D(fig2)
bx.scatter(C[:, 0], C[:, 1], C[:,2],c=cluster.labels_,cmap='rainbow')
bx.set_xlabel("petal Length")
bx.set_ylabel("petal width")
bx.set_zlabel("sepal width")
plt.show()

plt.figure(4,figsize=(10, 7))
dendrogram(Z,
            orientation='top',
            labels=cluster.labels_,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendogram Representation of clust Dataset")
plt.show()





