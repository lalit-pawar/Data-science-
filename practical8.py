#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns


# In[3]:


# Load the Titanic dataset
titanic = sns.load_dataset('titanic')


# In[4]:


titanic


# In[5]:


titanic.head


# In[6]:


titanic.describe


# In[7]:


titanic.isnull()


# In[8]:


sns.set_style('whitegrid')
sns.barplot(x='sex', y='survived', data=titanic)


# In[9]:


sns.set_style('whitegrid')
sns.barplot(x='class', y='survived', data=titanic)


# In[12]:


sns.set_style('whitegrid')
sns.boxplot(x='class', y='fare',data=titanic)


# In[ ]:




