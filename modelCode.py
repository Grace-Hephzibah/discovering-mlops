#!/usr/bin/env python
# coding: utf-8

# # Rent price in Barcelona 2014 - 2022
# ### A compilation of prices for rent in Barcelona Spain
# 
# # About Dataset
# This dataset includes data on price for rent in Barcelona, Spain. The data was collected for a period of 2014 - 2022 years, divided into trimesters.The prices go by neighbourhoods and districts.This dataset includes both prices per month and prices per square meter, so that you can easier compare them.
# 
# https://www.kaggle.com/datasets/marshuu/rent-price-in-barcelona-2014-2022

# In[1]:


# Importing the relevant Libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# The below code removes all the warnings during execution
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing the data from Kaggle 
df = pd.read_csv("data.csv")


# In[3]:


# Checking out the data 
df.head()


# In[4]:


# Checking the categorical unique values in "Year"
df["Year"].unique()


# In[5]:


# Checking the categorical unique values in "Trimester"
df["Trimester"].unique()


# In[6]:


# Checking the categorical unique values in "District"
df["District"].unique()


# In[7]:


# Checking the categorical unique values in "Average _rent"
df["Average _rent"].unique()


# In[8]:


# Checking the categorical unique values in "Neighbourhood"
df["Neighbourhood"].unique()


# In[9]:


# Checking if there are any null values 
# None are found!
df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


# Categorical values must be encoded to some constant numerical value 
## Simple technique is to use the map function 

## list(df["District"].unique()) -> gives the list of unique values 
## A list has index and element when enumerated. 
## The index itself is used as the numerical encoding. 

## Sorting helps it easier to find the code that corresponds to an element. 
## Consider using that for custom inputs 

# Encoding District 
df["District"] = df["District"].map({ele: index for index, ele in enumerate(list(df["District"].unique()))})

# Encoding Neighbourhood
df["Neighbourhood"] = df["Neighbourhood"].map({ele: index for index, ele in enumerate(list(df["Neighbourhood"].unique()))})

# Average _rent
df["Average _rent"] = df["Average _rent"].map({ele: index for index, ele in enumerate(list(df["Average _rent"].unique()))})


# In[12]:


# Checking the dataframe after encoding 
df.head()


# In[13]:


# Cross - verifying 
df.isnull().sum()


# In[14]:


# Checking if the encoded values are numerical 
df.info()


# In[15]:


# Creating the train test split 
# Usual rule: 80-20

# Importing the relevant library 
from sklearn.model_selection import train_test_split

# Features (x) and Label (y)
x = df.drop(columns = "Price", axis = 1)
y = df["Price"]

# The split (train - test )
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# # Feature Importance Plot 

# In[18]:


tc = df.corr()
val = sns.heatmap(tc)
fig = val.get_figure()
fig.savefig("out.png") 


# # Evaluation Metrics - Preprepared 

# In[19]:


# Create evaluation function (the competition uses Root Mean Square Log Error)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate our model
def show_scores(model):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_test, test_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_test, test_preds),
              "Training R^2": model.score(X_train, y_train),
              "Valid R^2": model.score(X_test, y_test)}
    return scores


# # About Random Forest Regressor 
# 
# ## Documentation Link 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# # ML Model - Random Forest Regressor 
# 
# Created and tested 3 models with different criterion. They are 
# 1. absolute error 
# 2. poisson 
# 3. squared error 

# In[20]:


from sklearn.ensemble import RandomForestRegressor


# In[21]:


model = RandomForestRegressor(
                    n_jobs=-1,
                    n_estimators = 500, 
                    criterion = "squared_error", 
                    max_samples = 3000
                    )
model.fit(X_train, y_train)


# In[22]:


metrics = show_scores(model)
metrics


# In[23]:


with open("metrics.txt", 'w') as outfile:
    for key in metrics:
        val = metrics[key]
        text = key + " : " + str(val)
        outfile.write(text)
        outfile.write("\n")


# # Model Successfully Created ✨
# 
# # Upvote this if you find useful!
