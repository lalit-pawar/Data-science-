#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:



# Download the Iris dataset and load it into a DataFrame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(url, names=column_names)


# In[3]:



# 1. List down the features and their types
feature_types = iris_df.dtypes
print("Features and their types:")
print(feature_types)


# In[4]:



# 2. Create a histogram for each feature
iris_df.hist(figsize=(10, 8))
plt.suptitle("Histograms of Iris Dataset Features")
plt.tight_layout()
plt.show()


# In[5]:



# 3. Create a box plot for each feature
plt.figure(figsize=(10, 8))
iris_df.boxplot()
plt.title("Box Plots of Iris Dataset Features")
plt.xticks(rotation=45)
plt.show()


# In[7]:





# In[ ]:




