# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:11:31 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics 
from scipy.stats import pearsonr

# Given That
File1=np.array([0.1, 0.2, 0.2, 0.2, 0.2 ,0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7,0.8, 0.8, 0.8, 0.9, 0.9, 0.8, 0.8, 0.9])
File2=np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.6, 0.5, 0.6, 0.7, 0.8, 0.8, 0.1, 0.2, 0.2, 0.6, 0.2 ,0.3, 0.4, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])
File3=np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.6, 0.5, 0.6, 0.7, 0.1, 0.7, 0.1, 0.2, 0.3, 0.6, 0.2 ,0.3, 0.4, 0.8, 0.1, 0.9, 0.4, 0.4, 0.4])
File4=np.array([0.1, 0.4, 0.4, 0.4, 0.4, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.9])
File5=np.array([0.1, 0.4, 0.4, 0.4, 0.4, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.9])

#a
File=[File1,File2,File3,File4,File5]

###############################################
def Mean(n_num):
    n = len(n_num) 
    get_sum = sum(n_num) 
    mean = get_sum / n 
    return mean
def MEAN(File):
    l=len(File)
    m=0
    all_means=np.array([])
    for i in range(l):
       m=Mean(File[i])
       all_means=np.append(all_means,m)
    return all_means  
All_MEAN_Values=MEAN(File)
print("Mean of all Features :")
print(All_MEAN_Values)
print("\n")

###############################################
def Mode(n_num):
    mode=statistics.mode(n_num)
    return mode
def MODE(File):
    l=len(File)
    m=0
    all_modes=np.array([])
    for i in range(l):
       m=Mode(File[i])
       all_modes=np.append(all_modes,m)
    return all_modes
All_Mode_Values=MODE(File)
print("Mode of all Features :")
print(All_Mode_Values)
print("\n")

###############################################
def Min(array):
    Min=np.array([])
    m=0
    a=len(array)
    for i in range(a):
        m=min(array[i])
        Min=np.append(Min,m)
    return Min
All_Minimum_Values=Min(File)
print("Minimum Value of all Features :")
print(All_Minimum_Values)
print("\n")

###############################################
def Max(array):
    Max=np.array([])
    m=0
    a=len(array)
    for i in range(a):
        m=max(array[i])
        Max=np.append(Max,m)
    return Max

All_Miximum_Values=Max(File)
print("Maximum Value of all Features :")
print(All_Miximum_Values)
print("\n")

Tabular_Table=pd.DataFrame()
Tabular_Table["Mean"]=All_MEAN_Values
Tabular_Table["Mode"]=All_Mode_Values
Tabular_Table["Min"]=All_Minimum_Values
Tabular_Table["Max"]=All_Miximum_Values

print(Tabular_Table)

##############################################
#B

# Low Variance Filter
###############################################

def Low_Variance_Filter(DataFrame):
    for feature in DataFrame.columns:
        print(np.var(DataFrame[feature]))
            
print("\n")
print("Variance :")
Low_Variance_Filter(Tabular_Table)

# Correlation

print("\n")
print("Correlation of Reduced Features :")
def Correlation(a,b):
    Correlation,_=pearsonr(a,b)
    return Correlation

print(Correlation(Tabular_Table["Mean"],Tabular_Table["Mode"]))

###############################################
Reduced_Features=pd.DataFrame()
Reduced_Features["Mean"]=Tabular_Table["Mean"]
Reduced_Features["Mode"]=Tabular_Table["Mode"]
print("\n")
print("Reduced Features :")
print(Reduced_Features)
# Plotting
###############################################
def Plot_Reduced_Features(a,b):
    plt.figure(1)
    plt.plot(a,'g')
    plt.plot(b,'r')
    plt.title("Reduced Features")
    plt.grid()

Plot_Reduced_Features(Tabular_Table["Mode"],Tabular_Table["Mean"])
















