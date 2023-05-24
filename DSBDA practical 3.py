#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statistics as st


# In[4]:


# Load the data
df=pd.read_csv("D:\College Practicals\DSBDApractical3\Loan_Default.csv")


# In[6]:


df.shape


# In[7]:


df.info()


# In[7]:


df.mean


# In[8]:


print(df.loc[:,'year'].median())


# In[9]:


print(df.loc[:,'income'].median())


# In[10]:


df.median(axis=1)[0:5]


# In[22]:


df.mode()


# In[23]:


df.std()


# In[24]:


print(df.loc[:,'year'].std())


# In[25]:


print(df.loc[:,'year'].std())


# In[26]:


df.std(axis=1)[0:5]


# In[27]:


df.var()


# In[28]:


from scipy.stats import iqr
iqr(df['income'])


# In[29]:


print(df.skew())


# In[30]:


df.describe()


# In[31]:


df.describe(include='all')


# In[ ]:




