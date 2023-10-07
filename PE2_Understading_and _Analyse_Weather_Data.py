# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:08:54 2020

@author: M.Samiuddin Rafay
"""
# importing neccessary libraries
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
import numpy as np
import pandas as pd

# Given Data
Ts1=[2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34]
Ts2=[-0.12, -0.16, -0.13, 0.28, 0.37, 0.39, 0.18, 0.09, 0.15, -0.06, 0.06, -0.07, -0.13, -0.18, -0.26]

#Converting given data to series
Ts1=pd.Series(Ts1)
Ts2=pd.Series(Ts2)

# creating an empty array of size Ts1 
a=np.zeros_like(Ts1)

#  figure 1
plt.figure(1, figsize=(10,20))

# Calculating Euclidean distance of given data series and Saving into empty array  
for i in range(0,15):
    a[i]=distance.euclidean(Ts1[i],Ts2[i])
    #print(a[i])

# Plotting both Series and their Euclidean distance using subplot
plt.subplot(211)
plt.plot(Ts1,'-o',color='blue')
plt.plot(Ts1)
plt.plot(Ts2,'-o',color='red')
plt.plot(Ts2)
plt.grid()
plt.legend(["Ts1","Ts2"], loc ="upper right")
plt.subplot(212)
plt.plot(a,'-o',color='green')
plt.grid()
plt.legend(["Ed1"], loc ="upper right")

#  figure 2
plt.figure(2,figsize=(8,6))

# creating an empty array of size Ts1 
e=np.zeros_like(Ts1)

# Calculating Zscores and Euclidean distance of Zscores of given data series and Saving into empty array  
for i in range(0,15):
    TNs1=stats.zscore(Ts1)
    TNs2=stats.zscore(Ts2)
    e[i]=distance.euclidean(TNs1[i],TNs2[i])
    print(e[i])

# Plotting  Zscores and Euclidean distance of Zscores of given data series
#print(TNs1)

#print(TNs2)
plt.subplot(211)
plt.plot(TNs1)
plt.plot(TNs1,'-o',color='blue')
plt.plot(TNs2,'-o',color='red')
plt.grid()
plt.legend(["TNs1","TNs2"], loc ="upper right")
plt.subplot(212)
plt.plot(e,'-o',color='green')
plt.grid()
plt.legend(["Ed2"], loc ="upper right")
plt.show()

# Comparing Plots of EDI and ED2

#  figure 3
plt.figure(3,figsize=(6,4))
plt.subplot(121)
plt.plot(a,'-o',color='blue')
plt.grid()
plt.legend(["Ed1"], loc ="upper right")
plt.subplot(122)
plt.plot(e,'-o',color='green')
plt.grid()
plt.legend(["Ed2"], loc ="upper right")


