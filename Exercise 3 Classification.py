#!/usr/bin/env python
# coding: utf-8

# # Classification Exercise - SVM

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[5]:


df_train = pd.read_csv('LoanProfile-training.csv')
df_train


# In[6]:


df_test = pd.read_csv('LoanProfile-test.csv')
df_test


# In[7]:


def myfunction(x):
    if x == 'N':
        return 0
    else:
        return 1 


# In[8]:


df_train['Default'] = df_train['Default'].apply(myfunction)
df_train


# In[9]:


df_test['Default'] = df_test['Default'].apply(myfunction)
df_test


# In[10]:


df_0 = df_train[df_train['Default'] == 0]
df_1 = df_train[df_train['Default'] == 1]

fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(df_0['Age'], df_0['Creditdebt'])
ax.scatter(df_1['Age'], df_1['Creditdebt'])


# In[11]:


X_train = df_train.iloc[:,0:2]
X_test = df_test.iloc[:,0:2]
y_train = df_train['Default']
y_test = df_test['Default']


# In[12]:


from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)


# In[13]:


y_pred= svc.predict(X_test)
print(y_pred)


# In[14]:


print(svc.predict([[35, 2.7777]]))


# In[15]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

