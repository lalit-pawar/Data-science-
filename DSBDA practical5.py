#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv("D:\College Practicals\DSBDApractical5\Social_Network_Ads.csv")


# In[3]:


df


# In[4]:


df.isnull()


# In[5]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])


# In[6]:


df


# In[7]:


# Select the columns you want to include in the covariance matrix
columns = ['Age', 'EstimatedSalary', 'Purchased']

# Create a new DataFrame that contains only the selected columns
df_selected = df[columns]

# Build the covariance matrix
covariance_matrix = df_selected.cov()

# Print the covariance matrix
print(covariance_matrix)


# In[8]:


# Select the independent variables
X = df.drop('Purchased', axis=1)

# Select the dependent variable
Y = df['Purchased']


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


# Select the independent and dependent variables
X = df.drop('Purchased', axis=1)
Y = df['Purchased']


# In[22]:



# Split the dataset into a training set and a testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[23]:


#import the class
from sklearn.linear_model import LogisticRegression


# In[24]:


#instantiate the model 

logreg = LogisticRegression()


# In[25]:


logreg.fit(X_train,Y_train)


# In[30]:


y_pred=logreg.predict(X_test)


# In[32]:


print("Predicted values for the testing data: ", Y_pred)


# In[27]:


# Predict the Y values for the training data
Y_pred = logreg.predict(X_train)


# In[29]:


print("Predicted values for the training data: ", Y_pred)


# In[35]:


from sklearn.metrics import accuracy_score

# Generate predicted values for the training data
Y_pred = logreg.predict(X_train)


# In[36]:



# Calculate the accuracy of the model
train_accuracy = accuracy_score(Y_train, Y_pred)


# In[37]:



# Print the accuracy
print("Accuracy on training data: {:.2f}%".format(train_accuracy * 100))


# In[40]:


from sklearn.metrics import precision_score,confusion_matrix,recall_score

# Calculate the precision score of the model
train_precision = precision_score(Y_train, Y_pred)


# In[41]:


print("Precision on training data: {:.2f}%".format(train_precision * 100))


# In[43]:


# Calculate the confusion matrix of the model
train_cm = confusion_matrix(Y_train, Y_pred)


# In[44]:


print("Confusion matrix on training data:\n", train_cm)


# In[46]:


# Calculate the recall score of the model
train_recall = recall_score(Y_train, Y_pred)


# In[47]:


print("Recall on training data: {:.2f}%".format(train_recall * 100))


# In[ ]:




