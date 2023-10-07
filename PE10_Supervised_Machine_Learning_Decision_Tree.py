# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:06:49 2021

@author: User
"""
# importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

# Loading iris datset

fisher_iris=pd.read_csv(r"C:\Users\User\Downloads\iris.data")

# Before implementing Decision Tree on fisher_iris Data frame lets do some data quality assessment

# Converting fisher_iris as a Data Frame 

fisher_iris=pd.DataFrame(fisher_iris)

# Printing the head of fisher_iris Data Frame

print("Head of fisher_iris: \n", fisher_iris.head())
print("\n")
# Spcifying the columns name

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
fisher_iris.columns = attributes

# Printing shape/ Dimentions fisher_iris  of Data Frame

print("Head of fisher_iris: \n",fisher_iris.shape)
print("\n")
# Printing statistics of fisher_iris  Data Frame

print("Head of fisher_iris: \n", fisher_iris.describe())
print("\n")
#plotting the decision tree for iris dataset

plt.figure(1)

# Slicing Train and Target Data

X, y = load_iris(return_X_y=True)

# Using Decision Tree Classifier 

clf = tree.DecisionTreeClassifier()

# Fitting target and Train Data

clf = clf.fit(X, y)

# plotting Decision Tree

tree.plot_tree(clf)

anypoint=[[3., 2., 3, 4]]

# predicting

prediction=clf.predict(anypoint)
print("\n")
print("Prediction of any point: \n",prediction)



