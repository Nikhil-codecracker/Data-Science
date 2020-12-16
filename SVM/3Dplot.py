#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[16]:


X,Y = make_circles(n_samples=500,noise=0.052)


# In[17]:


print(X.shape,Y.shape)


# In[18]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[19]:


def phi(X):
    
    """Non Linear Transformation"""
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X1**2+X2**2
    
    X_ = np.zeros((X.shape[0],3))
    print(X_.shape)
    
    X_[:,:-1] = X
    X_[:,-1] = X3
    
    return X_
    


# In[20]:


X_ = phi(X)


# In[21]:


print(X[:3,:])


# In[22]:


print(X_[:3,:])


# In[41]:


def plot3d(X,show=True):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    
    if(show==True):
        plt.show()
    return ax


# In[42]:


ax = plot3d(X_)


# ### Logistic Classifier

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[27]:


lr = LogisticRegression()


# In[28]:


acc = cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f"%(acc*100))


# ### Logistic Classifier on Higher Dimension Space 

# In[29]:


acc = cross_val_score(lr,X_,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f"%(acc*100))


# ### Visualise the Decision Surface 

# In[30]:


lr.fit(X_,Y)


# In[35]:


wts = lr.coef_


# In[36]:


bias = lr.intercept_


# In[37]:


xx,yy = np.meshgrid(range(-2,2),range(-2,2))
print(xx)
print(yy)


# In[38]:


z = -(wts[0,0]*xx + wts[0,1]*yy+bias)/wts[0,2]
print(z)


# In[40]:





# In[44]:


ax = plot3d(X_,False)
ax.plot_surface(xx,yy,z,alpha=0.5)
plt.show()


# In[ ]:




