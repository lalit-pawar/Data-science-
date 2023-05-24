#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv("D:\College Practicals\DSBDApractical2\StudentPerformance.csv")


# In[6]:


dataset


# In[3]:


dataset.isnull()


# In[11]:


series= pd.isnull(dataset["math score"])


# In[12]:


dataset[series]


# In[13]:


dataset.notnull()


# In[14]:


series1 = pd.notnull(dataset["math score"])


# In[15]:


dataset[series1]


# In[4]:


missing_values = ["Na", "na"]
dataset=pd.read_csv("D:\College Practicals\DSBDApractical2\StudentPerformance.csv", na_values= missing_values)


# In[5]:


dataset


# In[20]:


ndf=dataset
ndf.fillna(0)


# In[21]:


m_v = dataset ['math score'].mean()
dataset['math score'].fillna(value= m_v , inplace= True)


# In[22]:


dataset


# In[23]:


ndf.replace(to_replace = np.nan, value = -99)


# In[24]:


ndf.dropna()


# In[25]:


ndf.dropna(how= 'all')


# In[26]:


ndf.dropna(axis= 1)


# In[27]:


#Boxploting
import matplotlib.pyplot as plt

col = ['math score' , 'reading score' , 'writing score' , 'placement score']


# In[28]:


dataset.boxplot(col)


# In[29]:


print(np.where(dataset['reading score']> 90))


# In[30]:


print(np.where(dataset['writing score']<70))


# In[31]:


# Detecting Outliers

import matplotlib.pyplot as plt


# In[32]:


fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(dataset['placement score'], dataset['placement offer count'])


# In[33]:


plt.show()


# In[34]:


ax.set_xlabel('(proportion non-retail business acres )/(town)')


# In[35]:


ax.set_ylabel('(Full-value property-tax rate )/($10,000)')


# In[36]:


print(np.where((dataset['placement score']<50)  & (dataset['placement offer count']>1)))


# In[37]:


print(np.where((dataset['placement score']>85)  & (dataset['placement offer count']<3)))


# In[38]:


# Detecting Outliers using Z-score

from scipy import stats


# In[39]:


z = np.abs(stats.zscore(dataset['math score']))


# In[40]:


print(z)


# In[41]:


# To define ooutlier threshold value is chosen

threshold = 0.18


# In[42]:


sample_outliers = np.where(z<threshold)


# In[43]:


sample_outliers 


# In[44]:


# Detecting Outliers using (IQR)


# In[45]:


sorted_rscore = sorted(dataset['reading score'])


# In[46]:


sorted_rscore


# In[47]:


q1 = np.percentile(sorted_rscore, 25)
q3 = np.percentile(sorted_rscore, 75)


# In[48]:


print(q1,q3)


# In[49]:


IQR = q3-q1


# In[50]:


lwr_bound = q1-(1.5*IQR)
upr_bound = q3+ (1.5*IQR)


# In[51]:


print(lwr_bound, upr_bound)


# In[52]:


r_outliers =[]


# In[53]:


for i in sorted_rscore:
    if(i<lwr_bound or i>upr_bound):
        r_outliers.append(i)
        
print(r_outliers )        


# In[54]:


# Trimming and removing the outliers 


# In[55]:


new_df = dataset

for i in sample_outliers:
    new_df.drop(i,inplace=True)
    


# In[56]:


new_df


# In[57]:


#Flooring and capping 


# In[58]:


df_stud = dataset
ninetieth_percentile = np.percentile(df_stud['math score'],90)


# In[59]:


b = np.where(df_stud['math score']>ninetieth_percentile, ninetieth_percentile, df_stud['math score'])


# In[60]:


print("New array:" , b)


# In[61]:


df_stud.insert(1,"m score", b, True)


# In[62]:


df_stud


# In[63]:


# Mean/Median imputation:
import matplotlib.pyplot as plt

col = ['reading score']
dataset.boxplot(col)


# In[64]:


median = np.median(sorted_rscore)
median


# In[65]:


redefined_df = dataset

redefined_df['reading score'] =np.where(redefined_df['reading score']>upr_bound,median,redefined_df['reading score'])

redefined_df


# In[66]:


redefined_df = dataset

redefined_df['reading score'] =np.where(redefined_df['reading score']<lwr_bound,median,redefined_df['reading score'])

redefined_df


# In[67]:


import matplotlib.pyplot as plt

col = ['reading score']


redefined_df.boxplot(col)


# In[68]:


import matplotlib.pyplot as plt

new_df['math score'].plot(kind = 'hist')
dataset['log math'] = np.log10(dataset['math score'])
dataset['log math'].plot(kind = 'hist')


# In[ ]:





# In[ ]:




