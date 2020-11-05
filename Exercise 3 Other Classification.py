#!/usr/bin/env python
# coding: utf-8

# # Other Classification Exercise - KNN
# Peform the same exercise as Logistic Regression using KNN Classifier
# In[2]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("iris-data-clean.csv")
df['class']


# In[4]:


df1 = df
df1.loc[df['class'] == 'Setosa', 'class'] = 0
df1.loc[df['class'] == 'Virginica', 'class'] = 1
df1.loc[df['class'] == 'Versicolor', 'class'] = 2
df1 = df1.astype({'class': int})
df1['class']


# In[5]:


from sklearn.model_selection import train_test_split

X = df1.iloc[:, 0:4]
y = df1['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)


# In[17]:


y_pred = knn.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[19]:


X1 = df1.iloc[:, 2:4]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, random_state = 42)


# In[20]:


knn.fit(X1_train, y1_train)

y1_pred = knn.predict(X1_test)
print(accuracy_score(y1_test, y1_pred))

