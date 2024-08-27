# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:55:40 2024

@author: Gaurav Bombale
"""
"""
A cloth manufacturing company is interested to know about
the different attributes contributing to high sales. 
Build a decision tree & random forest model with Sales 
as target variable (first convert it into categorical variable).
"""
'''
1. Business Problem

1.1. Business Objective:
The primary objective is to identify and understand the various attributes that significantly contribute to high sales for a cloth manufacturing company. To achieve this goal, we aim to build both a decision tree and a random forest model. The target variable for these models will be 'Sales,' which will be converted into a categorical variable.

1.2. Constraints:

Availability and quality of data: The accuracy and reliability of the models will depend on the quality and quantity of the dataset provided.
Interpretability: The decision tree and random forest models are chosen for their ability to provide insights into feature importance, but ensuring the interpretability of the results is crucial.
2. Data Dictionary for the Given Dataset

The dataset comprises the following columns:

Sales (Target Variable): Categorical variable indicating the level of sales.
CompPrice: The price charged by the company for its products.
Income: The average income of the customers in the area.
Advertising: The advertising budget for the products.
Population: The population size in the area where the products are sold.
Price: The price at which the products are sold.
ShelveLoc: The quality of the shelving location, categorized as 'Good,' 'Medium,' or 'Bad.'
Age: The average age of the customers in the area.
Education: The education level of the customers, measured on a scale.
Urban: Binary variable indicating whether the store is located in an urban area (Yes/No).
US: Binary variable indicating whether the store is located in the United States (Yes/No).
'''

"""
Business Objective 
Minimize : Minimize costs of cloths.
Maximaze : Maximize overall Sales.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Company_Data.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()

df.shape
# 400 rows and 11 columns
df.columns
'''
['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 no null values

import seaborn as sns
import matplotlib.pyplot as plt
# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# boxplot
# boxplot on Sales column
sns.boxplot(df.Sales)
# In Sales column 2 outliers 

sns.boxplot(df.Income)
# In Income column no outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['Income'],kde=True)
# normally right skew and the distributed

sns.histplot(df['Sales'],kde=True)
# skew and the distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df.dtypes
# Some columns in int, float data types and some Object

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# sum is 0.

df.isnull().sum()
df.dropna()
df.columns

data=df[::]

# Converting into binary
lb=LabelEncoder()
data["SheveLoc"]=lb.fit_transform(data["ShelveLoc"])
data["US"] = lb.fit_transform(data["US"])
data["Urban"]= lb.fit_transform(data["Urbon"])

data["US"].unique()
data['US'].value_counts()
colnames=list(data.columns)

predictors=colnames[:10]
target=colnames[9]

# Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors], train[target])
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target], preds_test,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])

# Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target], preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])

# 100 % accuracy 
# Accuracy of train data > Accuracy test data i.e Overfit model


