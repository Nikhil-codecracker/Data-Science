#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


X = pd.read_csv('.\Hardwork Pays Off (Linear Regression)\Linear_X_Train.csv').values
Y = pd.read_csv('.\Hardwork Pays Off (Linear Regression)\Linear_Y_Train.csv').values


# In[9]:


theta = np.load(".\Hardwork Pays Off (Linear Regression)\Thetalist.npy")

T0 = theta[:,0]
T1 = theta[:,1]


# In[11]:


plt.ion() #To turn on the interactive mode
for i in range(0,50,3):
    y_pred=T1[i]*X+T0
    plt.scatter(X,Y)
    plt.plot(X,y_pred,'red')
    plt.draw()
    plt.pause(1)
    plt.clf()


# In[ ]:




