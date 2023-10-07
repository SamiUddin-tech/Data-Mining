# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:52:44 2020

@author: M.Samiuddin Rafay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance

# Task 3
user_input=float(input("Please Enter a number less than 1 :"))
print('\n')
index=int(input("Please Enter the index for new Entry :"))
def random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]
print('\n')
# Defining Threshold
threshold=0.5
hours=np.arange(0,20)

# Task 1

var=random_floats(0,1,20)
print("Health Condition : ")
print("\n")
print(var)

plt.figure(1,figsize=(10,8))
plt.scatter(hours,var)
plt.axhline(y=0.5, color='red')
plt.xlabel('hours')
plt.ylabel('Health Condition')
plt.grid()
health_condition=np.array([])

col=[]


# Task 2 
for i in var:
    if(i>=threshold):
        health_condition=np.append(health_condition,1)
    else:
        health_condition=np.append(health_condition,0)

print("\n")
print("After Discretization :")
print("Good : 0")
print("Faulty : 1")
print("\n")
print("health_condition :")
print("\n")
print(health_condition)

for i in health_condition:
    if (i==1):
        col.append('r')
    elif(i==0):
        col.append('b')
    
# Task 4 Plot

for i in range(20):
    plt.figure(2, figsize=(8,6))
    plt.scatter(hours[i],var[i], c=col[i])
    plt.xlabel('Time in Hours')
    plt.ylabel('Health feature')
    #plt.axhline(y=0.5, color='black')
plt.grid()

var_series=pd.Series(var)
distance_vector=np.zeros_like(var_series)
#print('\n')
#print('Var_Series : ')
#print(var_series)
# task 4 Plot
for i in range(0,20):
    distance_vector[i]=distance.euclidean(var_series[i], user_input)
plt.figure(3, figsize=(6,4))
plt.plot(distance_vector, 'o-')
plt.title("Distance Feature")
plt.xlabel("Euclidean distance")
plt.ylabel("Hour")
plt.grid()

distance_vector=np.array(distance_vector)
# printing Distance Vector
print("\n")
print("Distance Vector :")
print("\n")
print(distance_vector)

# Task 5
indexes_of_distance=np.array([])
index_state=np.array([])
indexes_of_distance=np.append(indexes_of_distance,np.where(var_series<threshold))

for i in indexes_of_distance:
    index_state=np.append(index_state,var_series[i])

print("\n")
print("Indexes at which D vector <= threshold")
print("\n")
print(indexes_of_distance)
print('\n')
print("Corresponding states on the same indexes from state feature :")
print("\n")
print(index_state)


# Task 6
# Use mode command to determine most similar class
from collections import Counter
most_similar_class=Counter(health_condition)
most_similar_class=most_similar_class.most_common(1)
print("\n")
print("The Most Similar class : ")
print("\n")
print(most_similar_class)
print("\n")
# Condition of New Data Point
if (user_input>=threshold):
    user_entry=1
    print("New Entry is Faulty")
else:
    user_entry=0
    print("New Entry is Good")

print("\n")

plt.figure(4, figsize=(4,4))

sorting_var=np.array([])
for i in range(20):
    sorting_var=np.append(sorting_var,var[i])
sorted_var=sorted(sorting_var)
print('\n')
print('Sorted Health Features : ')
print('\n')
print(sorted_var)


color=[]
for i in sorted_var:
    if (i>=threshold):
        color.append('r')
    else:
        color.append('b')
for i in range(20):
    plt.scatter(hours[i],sorted_var[i], c=color[i])
    plt.plot(index,user_input, 'x', color='purple', markersize=20)
    plt.xlabel('Time in Hours')
    plt.ylabel('Health feature')
    #plt.axhline(y=0.5, color='black')
plt.grid()


# Discritization of sorted_Health conditions
disc_sorted_var=np.array([])
for i in sorted_var:
    if(i>=threshold):
        disc_sorted_var=np.append(disc_sorted_var,1)
    else:
        disc_sorted_var=np.append(disc_sorted_var,0)
# plotting

plt.figure(5, figsize=(4,3))
plt.plot(disc_sorted_var)
plt.xlabel('Hour')
plt.ylabel('State')
