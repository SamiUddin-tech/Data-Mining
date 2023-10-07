# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:45:15 2020

@author: M.Samiuddin Rafay
"""

# importing neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# reading "weatherHistory.csv" using read_csv under pandas
Weather_Dataset=pd.read_csv("D:\weatherHistory.csv")
# printing head of Weather_Dataset
print("\n")
print("Head of Weather_Dataset")
print(Weather_Dataset.head())
# storing each column of Weather_Dataset in a separate variable
Temperature = Weather_Dataset["Apparent Temperature (C)"]
Humidity = Weather_Dataset["Humidity"]
Wind_Speed= Weather_Dataset["Wind Speed (km/h)"]
Wind_Bearing = Weather_Dataset["Wind Bearing (degrees)"]
Visibility = Weather_Dataset["Visibility (km)"]
Loud_Cover = Weather_Dataset["Loud Cover"]
Pressure = Weather_Dataset["Pressure (millibars)"]
D_Summary= Weather_Dataset["Daily Summary"]
print("\n")

print("Correlation Coefficient between Temperature and Humidity")
print(np.corrcoef(Temperature, Humidity))
print("\n")
print("Correlation Coefficient between Temperature and Wind_Speed")
print(np.corrcoef(Temperature, Wind_Speed))
print("\n")
print("Correlation Coefficient between Temperature and Wind_Bearing")
print(np.corrcoef(Temperature, Wind_Bearing))
print("\n")
print("Correlation Coefficient between Temperature and Visibility")
print(np.corrcoef(Temperature, Visibility))
print("\n")
print("Correlation Coefficient between Temperature and Loud_Cover")
print(np.corrcoef(Temperature, Loud_Cover))
print("\n")
print("Correlation Coefficient between Temperature and Pressure")
print(np.corrcoef(Temperature, Pressure))
print("\n")

# Sub-plotting all correlations with temperatures using scatter plot
plt.figure(2, figsize=(20,10))

plt.subplot(241)
plt.plot(Temperature[0:365],color='orange')
plt.grid()
plt.title("Temperature")

plt.subplot(242)
plt.scatter(Temperature[0:365],Humidity[0:365],color='red')
plt.grid()
plt.title("Temperature Vs. Humidity")

plt.subplot(243)
plt.scatter(Temperature[0:365],Wind_Speed[0:365],color='green')
plt.grid()
plt.title("Temperature Vs. Wind Speed")

plt.subplot(244)
plt.scatter(Temperature[0:365],Wind_Bearing[0:365],color='blue')
plt.grid()
plt.title("Temperature Vs. Wind Bearing")

plt.subplot(245)
plt.bar(Temperature[0:365],D_Summary[0:365],color='brown')
plt.title("Temperature Vs. Daily Summary")
plt.grid()

plt.subplot(246)
plt.scatter(Temperature[0:365],Loud_Cover[0:365],color='yellow')
plt.grid()
plt.title("Temperature Vs. Loud Cover")

plt.subplot(247)
plt.scatter(Temperature[0:365],Visibility[0:365],color='black')
plt.grid()
plt.title("Temperature Vs. Visibility")

plt.subplot(248)
plt.scatter(Temperature[0:365],Pressure[0:365],color='purple')
plt.grid()
plt.title("Temperature Vs. Pressure")