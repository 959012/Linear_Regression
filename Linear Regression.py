#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


# In[3]:


diabetes = datasets.load_diabetes()


# In[4]:


print(diabetes.keys())


# In[5]:


print(diabetes.target)


# In[6]:


print(diabetes.DESCR)




# In[7]:


#diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x = diabetes.data


# In[8]:


diabetes_x


# In[9]:


diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-20:]


# In[10]:


diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-20:]


# In[11]:


model = linear_model.LinearRegression()


# In[12]:


model.fit(diabetes_x_train,diabetes_y_train)


# In[13]:


diabetes_y_predit = model.predict(diabetes_x_test)


# In[14]:


print("Mean squard error is ", mean_squared_error(diabetes_y_test,diabetes_y_predit))


# In[15]:


print("weights:", model.coef_)
print("Intersept:", model.intercept_)


# In[ ]:




