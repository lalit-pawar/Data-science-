#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df= pd.read_csv("D:\College Practicals\DSBDApractical6\PlayTennis.csv")


# In[6]:


df


# In[7]:


df.isnull()


# In[8]:


ndf=df
ndf.fillna(0)


# In[9]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Temperature'] = le.fit_transform(df['Temperature'])


# In[10]:


df


# In[11]:


# Define the independent and dependent variables
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
Y = df['Play Tennis']


# In[12]:


# Print the first few rows of the X dataframe
print(X.head())


# In[13]:


# Print the first few rows of the Y dataframe
print(Y.head())


# In[14]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[22]:


get_ipython().system('pip install --upgrade pandas')


# In[23]:


X = df.get_dummies(X, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])


# In[ ]:


# Print the shape of the training sets
print("X_train shape:", X_train.shape)


# In[ ]:


# Print the shape of the testing sets
print("X_test shape:", X_test.shape)


# In[ ]:


# Print the shape of the training sets
print("Y_train shape:", Y_train.shape)


# In[ ]:


# Print the shape of the testing sets
print("Y_test shape:", Y_test.shape)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

# Create an instance of the Gaussian Naive Bayes model
model = GaussianNB()

# Train the model using the training set
model.fit(X_train, Y_train)


# In[ ]:


# Use the model to make predictions on the testing set
Y_pred = model.predict(X_test)


# In[20]:


from sklearn.metrics import precision_score,confusion_matrix, accuracy_score,recall_score

accuracy= accuracy_score( Y_test, Y_pred)


# In[18]:


print("Accuracy:", accuracy)


# In[19]:


precision = precision_score(Y_test,Y_pred, average='micro')


# In[65]:


print("Precision:", precision)


# In[66]:


recall = recall_score(Y_test,Y_pred, average='micro')


# In[67]:


print("Recall:", recall)


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


cm = confusion_matrix(Y_test,Y_pred)


# In[70]:


print("Confusion matrix on training data:\n", cm)


# In[ ]:




