#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn import preprocessing


# In[5]:


#Importing the file 
df=pd.read_csv("D:\College Practicals\DSBDApractical1\spambase_csv.csv")


# In[6]:


df


# In[7]:


#data visualization
df.head(n=10)


# In[8]:


#Indexing 
df.index


# In[9]:


# TO get no of rows and columns
df.shape 


# In[10]:


#To check whether the target attribute is binary or not 
np.unique(df['class'])


# In[12]:


df.columns.values


# In[13]:


df['word_freq_address']


# In[14]:


df.isnull()


# In[16]:


df.iloc


# In[15]:


df.isnull().ne


# In[16]:



df.isnull().sum().sum()


# In[17]:


df.isnull().sum(axis=1)


# In[18]:


df.word_freq_remove.isnull().sum()


# In[19]:


df.groupby(['word_freq_address'])['word_freq_make'].apply(lambda x:x.isnull().sum())


# In[17]:


df.describe(include='all')


# In[21]:


df.sort_index(axis=1,ascending=False)


# In[22]:


df.sort_values(by="word_freq_address")


# In[23]:


df.iloc[5]


# In[24]:


df[0:3]


# In[25]:


df.loc[:,["word_freq_make","word_freq_address"]]


# In[26]:


df.iloc[:4,:]


# In[27]:


df.iloc[:, :4]


# In[28]:


df.iloc[:4, :6]


# In[29]:


df.iloc[3:5, 0:2]


# In[30]:


df.iloc[[1,2,4],[0,2]]


# In[31]:


df.iloc[3:5,:]


# In[32]:


df.iloc[:, 1:3]


# In[33]:


df.iloc[1:1]


# In[34]:


col_1_4=df.columns[1:4]


# In[35]:


df[df.columns[2:4]].iloc[5:10]


# In[36]:


min_max_scaler=preprocessing.MinMaxScaler()


# In[37]:


x=df.iloc[:,:4]


# In[38]:


x_scaled=min_max_scaler.fit_transform(x)


# In[39]:


df_normalized= pd.DataFrame(x_scaled)


# In[40]:


df_normalized


# In[18]:


df.head


# In[ ]:




