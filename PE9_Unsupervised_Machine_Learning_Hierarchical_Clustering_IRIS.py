# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:58:59 2021

@author: User
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

fisher_iris=pd.read_csv(r"C:\Users\User\Downloads\iris.data")
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
fisher_iris.columns = attributes

petal_length=fisher_iris["petal_length"]
petal_width=fisher_iris["petal_width"]
sepal_length=fisher_iris["sepal_length"]
sepal_width=fisher_iris["sepal_width"]
X=np.array(list(zip(petal_length,petal_width,sepal_width)))
Z = linkage(X[:,:2], 'ward')

labels = X
plt.figure(1,figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')
plt.xlabel("petal Length")
plt.ylabel("petal Width")
plt.show()
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom', fontsize=8)
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
print(cluster.labels_)
plt.figure(2,figsize=(10, 7))
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.xlabel("petal Length")
plt.ylabel("petal Width")
plt.show()

fig2 = plt.figure(4)
bx = Axes3D(fig2)
bx.scatter(X[:, 0], X[:, 1], X[:,2],c=cluster.labels_,cmap='rainbow')
bx.set_xlabel("petal Length")
bx.set_ylabel("petal width")
bx.set_zlabel("sepal width")
plt.show()

plt.figure(5,figsize=(10, 7))
dendrogram(Z,
            orientation='top',
            labels=cluster.labels_,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendogram Representation of Iris Dataset")
plt.show()

