#!/usr/bin/env python
# coding: utf-8

# # Parkinson's Disease Detection
# 

# Importing dependencies

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm	
from sklearn.metrics import accuracy_score


# Data collection and Analysis

# In[2]:


# load data from CSV to pandas dataframe
parkinsons_data = pd.read_csv('/Users/ajay_krsna/Documents/MyStuff/Code/Projects/parkinsons-diesease-detection/Data/parkinsons_data.csv')


# In[3]:


# print first 5 rows of dataframe
parkinsons_data.head()


# In[4]:


# number of rows and columns in the dataframe
parkinsons_data.shape


# In[5]:


# get more information about the dataset
parkinsons_data.info()


# In[6]:


# check for missing values 
parkinsons_data.isnull().sum()


# In[7]:


# get statistical information 
parkinsons_data.describe()


# In[8]:


# distribution of target variable --> column 'status'
parkinsons_data['status'].value_counts()


# 1 --> Parkinson's positive 
# 
# 0 --> Healthy 

# In[9]:


# group data based on the target variable
parkinsons_data.groupby('status').mean()


# Data Pre-Processing 

# In[10]:


# separating the features and target 

X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']


# In[11]:


print(X)


# In[12]:


print(Y)


# In[13]:


# split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# Data Standardization

# In[15]:


scaler = StandardScaler()


# In[16]:


scaler.fit(X_train)


# In[17]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Model Training: Support Vector Machine

# In[18]:


model = svm.SVC(kernel='linear')


# In[19]:


# train SVM with training data 
model.fit(X_train, Y_train)


# Model Evaluation:
# 

# In[20]:


# accuracy score on training data 

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data: ', training_data_accuracy)


# In[21]:


# accuracy score on test data 

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of training data: ', test_data_accuracy)


# Parkinson's Predictive system

# In[22]:


input_data = (214.28900,260.27700,77.97300,0.00567,0.00003,0.00295,0.00317,0.00885,0.01884,0.19000,0.01026,0.01161,0.01373,0.03078,0.04398,21.20900,0.462803,0.664357,-5.724056,0.190667,2.555477,0.148569)

# change input data into numpy array
input_data_as_np_array = np.asarray(input_data)

# reshape the numpy array
input_reshaped = input_data_as_np_array.reshape(1,-1)

# standardize the input data 
standard_data = scaler.transform(input_reshaped)

prediction = model.predict(standard_data)
print(prediction)

if prediction[0] == 0:
	print('The person is healthy')

else:
	print("The person has Parkinson's disease")


# # End Project
