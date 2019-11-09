#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"C:\Users\BISWA\Desktop\ML Project\Salary data.csv")


# In[3]:


data.info()


# In[4]:


data.columns


# In[5]:


data.head()


# In[6]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[12]:


y_test


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


model = LinearRegression()


# In[16]:


model.fit(x_train,y_train)


# In[19]:


y_pre = model.predict(x_test)


# In[20]:


y_pre


# In[22]:


y_test


# In[34]:


plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train,model.predict(x_train))


# In[35]:


from sklearn.metrics import mean_squared_error


# In[41]:


print("MSE is", mean_squared_error(y_test,y_pre))


# In[43]:


print("Weight is", model.coef_)
print("Intersh is:", model.intercept_)


# In[ ]:




