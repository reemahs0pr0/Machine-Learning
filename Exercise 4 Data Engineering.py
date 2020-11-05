#!/usr/bin/env python
# coding: utf-8

# # Data Engineering Exercise 
Refer to the your previous K-NN Classification workship using iris-data-clean.csv.  Modify the solution to load data from iris-data.csv.  

Note that the file contains some empty values in certain rows.  Does it affect how you train your machine learning model?  If yes, what action could you perform without switching to iris-data-clean.csv?
# In[2]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv("iris-data.csv")


# In[25]:


df1 = df
df1.loc[df['class'] == 'Setosa', 'class'] = 0
df1.loc[df['class'] == 'Virginica', 'class'] = 1
df1.loc[df['class'] == 'Versicolor', 'class'] = 2
df1 = df1.astype({'class': int})
df1


# In[10]:


from sklearn.model_selection import train_test_split

X = df1.iloc[:, 0:4]
y = df1['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# In[12]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[20]:


df2 = df1.dropna()
df2


# In[27]:


X1 = df2.iloc[:, 0:4]
y1 = df2['class']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state = 42)


# In[28]:


knn.fit(X1_train, y1_train)

y1_pred = knn.predict(X1_test)
print(accuracy_score(y1_test, y1_pred))

