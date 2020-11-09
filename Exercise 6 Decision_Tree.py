#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[2]:


df = pd.read_csv('iris-data-clean.csv')
df


# In[3]:


df1 = df
df1.loc[df['class'] == 'Setosa', 'class'] = 0
df1.loc[df['class'] == 'Virginica', 'class'] = 1
df1.loc[df['class'] == 'Versicolor', 'class'] = 2
df1 = df1.astype({'class': int})
df1['class']


# In[11]:


x = df1.iloc[:, 0:4]
y = df1['class']


# In[22]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42)


# In[23]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 3) 
dt.fit(x_train, y_train)


# In[24]:


y_pred = dt.predict(x_test)


# In[25]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[27]:


from sklearn import tree
import graphviz
from graphviz import Source

Source(tree.export_graphviz(dt, out_file=None, class_names=['Setosa', 'Virginica', 'Versicolor'] , feature_names= x_train.columns))

