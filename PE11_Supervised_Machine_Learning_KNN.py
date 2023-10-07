# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:38:50 2021

@author: User
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Loading iris datset

fisher_iris=pd.read_csv(r"C:\Users\User\Documents\IrIs.csv")

# Before implementing KNN on fisher_iris Data frame lets do some data quality assessment

# Converting fisher_iris as a Data Frame 

fisher_iris=pd.DataFrame(fisher_iris)

# Printing the head of fisher_iris Data Frame

print("Head of Data : \n",fisher_iris.head())

# Printing shape/ Dimentions fisher_iris  of Data Frame

print("\n")
print("Shape of Data : \n",fisher_iris.shape)

# Printing statistics of fisher_iris Data Frame

print("\n")
print("Statistics of Data : \n", fisher_iris.describe())

# Splitting the test and train Data

X = fisher_iris[["petal_length","petal_width"]]
y = fisher_iris[["species"]]

KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X,y)

# New Point

newpoint= [[5, 1.45]]

# New Point 2

newpoint2 = [[5, 1.45],[6,2],[2.75, 0.75]]

# Predicting New Point

Predict_NewPoint=KNN.predict(newpoint)

# Predicting New Point 2

Predict_NewPoint2=KNN.predict(newpoint2)

# Getting K Distances and indexes associated with 

Distances,Indexes=KNN.kneighbors(X=newpoint2, n_neighbors=5, return_distance=True)

# Printing Predicted Results
print("New Point 1 Prediction")
print(Predict_NewPoint)
print("New Point Prediction 2")
print(Predict_NewPoint2)
print("Distances")
print(Distances)
print("Indexes")
print(Indexes)


# plot for first point

plt.figure(1)
iris=sns.load_dataset('iris')
sns.scatterplot(x='petal_length',y='petal_width',data=iris,hue='species')
plt.plot(5,1.5,'x', color='k')
plt.grid(True)

# plot for Second point

plt.figure(2)
iris=sns.load_dataset('iris')
sns.scatterplot(x='petal_length',y='petal_width',data=iris,hue='species')
plt.plot(5,1.45,'x', color='k')
plt.plot(6,2,'x', color='k')
plt.plot(2.75,0.75,'x', color='k')
plt.grid(True)

plt.show()