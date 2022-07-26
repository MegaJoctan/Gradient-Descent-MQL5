# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:59:53 2022

@author: Omega Joctan
"""

#gradient descent implementation python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import seaborn as sns

from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


data = pd.read_csv(r'C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\892B47EBC091D6EF95E3961284A76097\MQL5\Files\Salary_Data.csv')

X = data["YearsExperience"].values.reshape(-1,1)
Y = data["Salary"].values.reshape(-1,1)

#data_scaled = np.array(scaler.fit_transform(data))

X = np.array(scaler.fit_transform(X))
Y = np.array(scaler.fit_transform(Y))


print(X,f"\n Scaled Mean {X.mean(axis=0)} std {X.std(axis=0)}")

plt.scatter(X, Y)
plt.xlabel("years experience")
plt.ylabel("Salary")
plt.show()


# Building the model

m = 0
c = 0

L = 0.1  # The learning Rate
epochs = 10000 # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 

for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

plt.figure(figsize=(12,6))
plt.title("Gradient Descent best model")
plt.scatter(X,Y)
plt.xlabel("Years Of Experince")
plt.ylabel("Salary")
plt.plot(X,m*X+c,label="Best model",c="red")
plt.legend(loc="upper right")
plt.show()



data = pd.read_csv(r'C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\892B47EBC091D6EF95E3961284A76097\MQL5\Files\titanic.csv')

print(data.head(5))

y = data.iloc[:,1]
x = data.iloc[:,2]

#print(x,"\n Y's ", y)


X = data["Pclass"].values.reshape(-1,1)
Y = data["Survived"].values.reshape(-1,1)

n = float(len(X))

e = 2.718281828; 

def Sigmoid(x):
    return 1/(1+(e**-x))

    
for i in range(epochs): 
    Y_pred = Sigmoid(m*X + c)  # The current predicted value of Y
    
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    
    if D_m == 0 and D_c == 0:
        break
    
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

sns.catplot( x="Sex", y = "Survived", kind="bar", hue="Pclass" , data=data)
sns.catplot(x="Pclass",y = "Survived", kind="swarm", hue="Sex", data=data)

Yp = np.sort(Sigmoid(m*X+c))



plt.figure(figsize=(13,9))
plt.scatter(X,Y, c = "blue")
plt.scatter(X,Yp, c="red",label ="sigmoid")
plt.legend(loc="upper right")
plt.show()

