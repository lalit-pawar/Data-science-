#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries and create alias for Pandas, Numpy and Matplotlibimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Import the Boston Housing dataset
from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()


# In[5]:


#Initialize the data frame
data = pd.DataFrame(boston.data)


# In[6]:


#Add the feature names to the dataframe
data.columns = boston.feature_names
data.head()


# In[7]:


#Adding target variable to dataframe
data['PRICE'] = boston.target


# In[8]:


# Perform Data Preprocessing( Check for missing values)
data.isnull().sum()


# In[9]:


#Split dependent variable and independent variables
x = data.drop(['PRICE'], axis = 1)
y = data['PRICE']


# In[10]:


#splitting data to training and testing dataset.
from sklearn.model_selection import train_test_split


# In[11]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2,random_state = 0)


# In[12]:


#Use linear regression( Train the Machine ) to Create Model
import sklearn
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model=lm.fit(xtrain, ytrain)


# In[14]:


#Predict the y_pred for all values of train_x and test_x
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)


# In[15]:


#Evaluate the performance of Model for train_y and test_y
df=pd.DataFrame(ytrain_pred,ytrain)
df=pd.DataFrame(ytest_pred,ytest)


# In[16]:


#Calculate Mean Square Paper for train_y and test_y
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(ytest, ytest_pred)
print(mse)


# In[17]:


mse = mean_squared_error(ytrain_pred,ytrain)
print(mse)


# In[18]:


#Plotting the linear regression model
plt.scatter(ytrain ,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred ,c='lightgreen',marker='s',label='Test data')


# In[19]:


plt.xlabel('True values')


# In[20]:


plt.ylabel('Predicted')


# In[21]:


plt.title("True value vs Predicted value")


# In[22]:


plt.legend(loc= 'upper left')


# In[23]:


#plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()


# In[ ]:





# In[ ]:




