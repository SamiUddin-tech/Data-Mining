# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 08:23:09 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from pyod.models.copod import COPOD

#from pca import pca

# Load the iris data from sklearn
iris = datasets.load_iris()
iris=pd.DataFrame(iris.data)
#print(iris)

# Get input variables (from column 1 to 4)

X=iris[iris.columns[0:4]]
#Y = iris[iris[:,:5]]
print(X)
#print(Y)

# De-mean data by subtrating mean form each point
X1=X-np.mean(X)
#print(X1)
#[COEFF, SCORE, LATENT,TSQUARED, EXPLAINED]=PCA.fit_transform(X1)


pca = PCA()
pca.fit(X1)
coeff = np.transpose(pca.components_)
print(coeff)

latent=(pca.explained_variance_)
print(latent)
#print(pca.score(X))
explained=pca.explained_variance_ratio_
print(explained*100)
#d=pca.singular_values_
#print(d)
f=pca.n_samples_
print(f)

numberOfDimentions=4;
reducedDimention=coeff[0,0:4]
print(reducedDimention)
reducedFeatureMatrix=X-reducedDimention
print(reducedFeatureMatrix)

plt.figure(3,figsize=(5,10))
X = np.arange(4)
col=['b','g','r','y']
for i in range(1,5,1):
#plt.hist(coeff[3])
    plt.subplot(2,2,i)
    plt.bar(X,coeff[i-1], color = col[i-1] )
    plt.title("Eigen Vector "+str(i))
    plt.grid()
plt.show()

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

