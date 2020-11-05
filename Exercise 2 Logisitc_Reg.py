#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression Exercise
# 1) Load iris datasets from iris-data-clean.csv
   # Replace the values in the columns 'Class' as follows:
     # "Setosa" = 0
     # "Virginica" = 1
     # "Versicolor" = 2
# 2) Using Logistic Regression, classify the outcome (Column : 'Class') based on the labels (Columns :'sepal length /cm', 'sepal width /cm', 'petal length /cm', 'petal width /cm')
    # a) Provide some values to predict the outcome
    # b) Validate the model - print the confusion matrix and the accuracy score
# 3) Redo the above steps with any two labels
    # a) Compare the accuracy score with the model built in the above with four features
# In[2]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("iris-data-clean.csv")
df1 = df.replace(["Setosa", "Virginica", "Versicolor"], [0, 1, 2])
df1


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logReg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')

x = df1[['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']]
y = df1['class'] 


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42)
x_train.head()


# In[13]:


logReg.fit(x_train, y_train)


# In[14]:


from sklearn.metrics import accuracy_score

y_pred = logReg.predict(x_test)
print(y_test)
print(y_pred)


# In[15]:


accuracy_score(y_test, y_pred)


# In[16]:


from sklearn.metrics import confusion_matrix
confusion_matrix (y_test, y_pred, labels = [1,0])


# In[17]:


print(logReg.predict([[10, 1.74, 7.5, 1.65]]))
print(logReg.predict([[7.5, 1.65, 10, 1.74]]))


# In[40]:


df_0 = df1.loc[df1['class'] == 0]
df_1 = df1.loc[df1['class'] == 1]
df_2 = df1.loc[df1['class'] == 2]
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.scatter(df_0['petal_length_cm'], df_0['petal_width_cm']) 
ax1.scatter(df_1['petal_length_cm'], df_1['petal_width_cm'], marker = 's') 
ax1.scatter(df_2['petal_length_cm'], df_2['petal_width_cm'], marker = 'x')
plt.show()


# In[41]:


x1 = df1[['petal_length_cm', 'petal_width_cm']]
y1 = df1['class'] 


# In[42]:


x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 42)
x1_train.head()


# In[43]:


logReg.fit(x1_train, y1_train)


# In[44]:


y1_pred = logReg.predict(x1_test)
print(y1_test)
print(y1_pred)


# In[45]:


accuracy_score(y1_test, y1_pred)


# In[46]:


confusion_matrix (y1_test, y1_pred, labels = [1,0])


# In[47]:


print(logReg.predict([[10, 1.74]]))
print(logReg.predict([[7.5, 1.65]]))

