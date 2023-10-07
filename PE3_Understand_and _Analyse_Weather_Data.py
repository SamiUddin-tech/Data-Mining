# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:12:25 2020

@author: M.Samiuddin Rafay
"""

# Feature extraction from Time Series Data

# importing neccessary python pakages

import matplotlib.pyplot as plt
import numpy as np
import random
import statistics
import pandas as pd


# defining function for generating random temperature values

def random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]
print('\n')


# defining array for storing respective values

Temperature_yearly=np.array([])
Temperature=np.array([])
Temperature_mean=np.array([])
Temperature_var=np.array([])


# generating Time Series Temperature, Mean per day, Variance per day
for i in range(365):
    Temperature=random_floats(0, 50, 24)
    Temperature_yearly=np.append(Temperature_yearly,Temperature)
    Temperature_mean=np.append(Temperature_mean ,np.mean(Temperature))
    Temperature_var=np.append(Temperature_var, statistics.variance(Temperature))


# printing shape of respective arrays to confirm the total values
print(Temperature_yearly.shape)
print(Temperature_mean.shape)
print(Temperature_var.shape)


# intializing figure 1 for plotting Temperature values

plt.figure(1, figsize=(10,8))


# plotting temperature per year

plt.subplot(111)
plt.plot(Temperature_yearly, color='purple')
plt.xlabel("Time (One Year)")
plt.ylabel("Temperature")
plt.grid()


# intializing figure 2 for plotting Mean Temperature values

plt.figure(2, figsize=(8,6))


# plotting Mean temperature per day

plt.subplot(111)
plt.plot(Temperature_mean, color='brown')
plt.xlabel("Time (365 Days)")
plt.ylabel("Mean(average) Temperature per Day")
plt.grid()


# intializing figure 2 for plotting Mean Temperature values

plt.figure(3, figsize=(6,4))

# intializing figure 2 for plotting Variance Temperature values

plt.subplot(111)
plt.plot(Temperature_var, color='gray')
plt.xlabel("Time (365 Days)")
plt.ylabel("Variance Temperature per Day")
plt.grid()

# showing all graphs

plt.show()


# Selecting any random value of variances as threshold

threshold=random.choice(Temperature_var)
print("\n")
print("Threshold : ")
print("\n")
print(threshold)


# defining an array for saving variance values greater than threshold 

days_above_threshold=np.array([])


# designing algorithm to sortout the day having variance values greater than Threshold 
 
for i in Temperature_var:
        days_above_threshold=np.append(days_above_threshold,np.where(Temperature_var>threshold))

# printing variances above threshold        
print("\n","Days above Threshold : ","\n")
print(days_above_threshold)


# converting days_above_threshold in DataFrame

day_above_DataFrame=pd.DataFrame(days_above_threshold)

# exporting days_above_threshold in .csv

day_above_csv=day_above_DataFrame.to_csv("D:\day_above_DataFrame.csv")




