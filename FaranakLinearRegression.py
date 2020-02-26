# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 08:55:52 2020

@author: faranak abri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
"""------------------------------------------------------"""
def closedForm(x,t):
    """
    w=np.dot((np.dot((np.linalg.pinv(np.dot(x.T,x))),x.T)),t)
    y=np.dot(x,w)
    """
    
    w=(np.linalg.pinv(x.T@x))@(x.T@t)
    y=x@w
    return y
"""------------------------------------------------------"""
def gradientDescend(x,t,lr=0.001,iteration=500,precision = 0.00000000001):
    w= np.random.rand(2,1)
    for i in range(iteration):
 #       gradient=(2*np.dot((np.dot(w.T,x.T)),x))-(2*np.dot(t.T,x))
        gradient=(2*((w.T@x.T)@x))-(2*(t.T@x))
        w0=w
        w=w0-(lr*gradient.T)
        if np.abs((w-w0)).all()<precision:
            break;
         
    y=np.dot(x,w) 
    return y
"""------------------------------------------------------"""

data=pd.read_excel("proj1Dataset.xlsx")
data = data[np.isfinite(data['Horsepower'])]

t=data['Horsepower']
t=(np.atleast_2d(t)).T

x=np.ones((len(t),2))
x[:,1]=data['Weight']
x_m=x.mean()
x_sd=x.std()
x_n=(x-x_m)/x_sd
    
y=closedForm(x_n,t)

plt.figure()
plt.subplot(121)
plt.title("Linear Regression-Closed Form")
plt.scatter(data['Weight'],data['Horsepower'],c='red',marker='+')
plt.xlabel("Weight")
plt.ylabel("Hordpower")
plt.plot(x[:,1],y,color='blue', label="Closed Form")
plt.legend(loc='upper right')

y=gradientDescend(x_n,t)

plt.subplot(122)
plt.title("Linear Regression-GradientDescend")
plt.scatter(data['Weight'],data['Horsepower'],c='red',marker='+')
plt.xlabel("Weight")
plt.ylabel("Hordpower")
plt.plot(x[:,1],y,color='g', label="Gradient Descend")
plt.legend(loc='upper right')

plt.show()
"""------------------------------------------------------"""

