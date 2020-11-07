#!/usr/bin/env python
# coding: utf-8

# # More Classification - Decision Tree

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[28]:


df_train = pd.read_csv('LoanProfile-training.csv')
df_test = pd.read_csv('LoanProfile-test.csv')


# In[29]:


df_train


# In[30]:


print(df_train['Age'].mean())
print(df_train['Creditdebt'].mean())


# In[31]:


def agefunction(age):
    if age > 36.7:
        return 1
    else:
        return 0
    
def credfunction(cred):
    if cred > 1.67:
        return 1
    else:
        return 0
    
def deffunction(x):
    if x == 'N':
        return 0
    else:
        return 1 
    
df_train['Age'] = df_train['Age'].apply(agefunction)
df_train['Creditdebt'] = df_train['Creditdebt'].apply(credfunction)
df_train['Default'] = df_train['Default'].apply(deffunction)
df_train


# In[32]:


df_test['Age'] = df_test['Age'].apply(agefunction)
df_test['Creditdebt'] = df_test['Creditdebt'].apply(credfunction)
df_test['Default'] = df_test['Default'].apply(deffunction)
df_test


# In[33]:


X_train = df_train.iloc[:,0:2]
X_test = df_test.iloc[:,0:2]
y_train = df_train['Default']
y_test = df_test['Default']


# In[35]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier() 
dt.fit(X_train, y_train) 


# In[37]:


y_pred = dt.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[40]:


from sklearn import tree
import graphviz
from graphviz import Source

Source(tree.export_graphviz(dt, out_file=None, class_names=['No', 'Yes'], feature_names= X_train.columns))

