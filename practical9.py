#!/usr/bin/env python
# coding: utf-8

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


# Load the Titanic dataset
titanic = sns.load_dataset('titanic')


# In[7]:


# Set the plotting style
sns.set_style('whitegrid')


# In[10]:


# Plot the box plot
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)


# In[14]:



# Set the title and labels
plt.title('Age Distribution by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')


# Display the plot
plt.show()


# In[13]:





# In[ ]:




