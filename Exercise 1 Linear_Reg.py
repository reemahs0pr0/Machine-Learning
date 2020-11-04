#!/usr/bin/env python
# coding: utf-8

# Linear Regression Example 
#1) Load the dataset 'OmniPower.csv'
#2) Create a Linear Regression Model with two independent variables (Price and Promotion), one dependent variable (Sale)
#3) Provide input values and predict the Sale
#4) What is the R square value?
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[100]:


df = pd.read_csv("OmniPower.csv")
df
df0 = df.groupby(["Price", "Promotion"]).mean("Sales")
df0.plot.bar()


#%%
df1 = pd.DataFrame ({
    'Price': [59,59,59,79,79,79,99,99,99],
    'Promotion': [200,400,600,200,400,600,200,400,600],
    'Sales': [df0.iloc[0,0], df0.iloc[1,0],df0.iloc[2,0],df0.iloc[3,0],
              df0.iloc[4,0], df0.iloc[5,0],df0.iloc[6,0],df0.iloc[7,0],
              df0.iloc[8,0]]
})
df1


# In[106]:

#various mathematical models

df1["x"] = np.log(df1.Promotion ** 0.5 / df1.Price ** 2)
# df1["x"] = np.log(df1.Promotion / df1.Price)
#df1["x"] = df1.Promotion / df1.Price ** 2
#df1["x"] = df1.Promotion ** 0.5 / df1.Price
#df1["x"] = df1.Promotion / df1.Price

plt.scatter(df1.x, df1.Sales)
plt.show()


# In[93]:


x_train = df1[["x"]]
y_train = df1.Sales


# In[94]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)


# In[95]:

print(lr.intercept_)
print(lr.coef_)
print(lr.score(x_train, y_train))


# In[105]:


price = float(input("Enter price: "))
promotion = float(input("Enter promotion: "))
print("Predicted sales: {0}".format(lr.predict([[np.log(promotion**0.5/price**2)]])))


# In[97]:


y_pred = lr.predict(x_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred, color='red')
plt.show()
