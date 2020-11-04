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
    
#%%

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

df = pd.read_csv("iris-data-clean.csv")
df1 = df.replace(["Setosa", "Virginica", "Versicolor"], [0, 1, 2])
df1

df_0 = df1[df1['class'] == 0]
df_1 = df1[df1['class'] == 1]
df_2 = df1[df1['class'] == 2]

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_0['sepal_length_cm'], df_0['sepal_width_cm'], df_0['petal_length_cm'], df_0['petal_width_cm']) 
ax.scatter(df_1['sepal_length_cm'], df_1['sepal_width_cm'], df_1['petal_length_cm'], df_1['petal_width_cm'], marker = 's') 
ax.scatter(df_2['sepal_length_cm'], df_2['sepal_width_cm'], df_2['petal_length_cm'], df_2['petal_width_cm'], marker = 'x')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logReg = LogisticRegression(solver = 'lbfgs')

x = df1[['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']]
y = df1['class'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
x_train.head()

logReg.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = logReg.predict(x_test)
print(y_test)
print(y_pred)

accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix (y_test, y_pred, labels = [1,0])

print(logReg.predict([[10, 1.74, 7.5, 1.65]]))
print(logReg.predict([[7.5, 1.65, 10, 1.74]]))

#%%

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.scatter(df_0['sepal_length_cm'], df_0['sepal_width_cm']) 
ax1.scatter(df_1['sepal_length_cm'], df_1['sepal_width_cm'], marker = 's') 
ax1.scatter(df_2['sepal_length_cm'], df_2['sepal_width_cm'], marker = 'x')
plt.show()

x1 = df1[['sepal_length_cm', 'sepal_width_cm']]
y1 = df1['class'] 

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 0)
x1_train.head()

logReg.fit(x1_train, y1_train)

y1_pred = logReg.predict(x1_test)
print(y1_test)
print(y1_pred)

accuracy_score(y1_test, y1_pred)

confusion_matrix (y1_test, y1_pred, labels = [1,0])

print(logReg.predict([[10, 1.74]]))
print(logReg.predict([[7.5, 1.65]]))
