# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:41:42 2024

@author: Gaurav Bombale
"""
"""
Divide the diabetes data into train and test datasets and 
build a Random Forest and Decision Tree model with Outcome 
as the output variable. 
"""
'''
1. Business Problem

1.1. Business Objective:
The objective is to develop predictive models using Random Forest and 
Decision Tree algorithms to understand the factors contributing to diabetes. 
The target variable for these models is 'Outcome.' The dataset will be divided 
into training and testing datasets for model evaluation.

1.2. Constraints:

Data quality: The accuracy and reliability of the models depend on the quality and 
completeness of the diabetes dataset.
Model interpretability: While Random Forest and Decision Tree models provide valuable 
insights, maintaining interpretability is crucial for understanding the factors 
influencing diabetes outcomes.
2. Data Dictionary for the Given Dataset

The dataset consists of the following columns:

Outcome (Target Variable): Binary variable indicating the presence (1) or absence (0) of diabetes.
Number of times pregnant: Number of pregnancies experienced by the individual.
Plasma glucose concentration: Concentration of glucose in the blood measured in milligrams per deciliter (mg/dL).
Diastolic blood pressure: Diastolic blood pressure level in mm Hg.
Triceps skin fold thickness: Thickness of the skin fold in millimeters.
2-Hour serum insulin: Insulin levels in the blood after a 2-hour period, measured in mU/mL.
Body mass index: BMI calculated as weight in kilograms divided by the square of height in meters.
Diabetes pedigree function: A function providing information about the diabetes hereditary risk.
Age (years): Age of the individual.
Class variable: An additional variable indicating the class label.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Diabetes.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()

df.shape
# 768 rows and 9 columns
df.columns
'''
[' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)', ' Class variable']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 no null values

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# Replace column names
# df.columns = df.columns.str.replace(' ', '_')
# df.columns

df.columns = [
    'Pregnant',
    'Glucose',
    'BloodPressure',
    'thickness',
    'insulin',
    'BMS',
    'Diabetes',
    'age',
    'class_variable'
]

# Now you can access the columns
print(df.columns)

# boxplot
# boxplot on Pregnant column
sns.boxplot(df.Pregnant)
# In Pregnant column 3 outliers 

sns.boxplot(df.age)
# In Income column many outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['age'],kde=True)
# normally right skew and the distributed

sns.histplot(df['Pregnant'],kde=True)
# right skew and the distributed

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
# Normalize data 
# Normalize the data using norm function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Apply the norm_fun to data 
df1=norm_fun(df.iloc[:,:8])

df['class_variable']
df1['class_variable']=df['class_variable']

df.isnull().sum()
df.dropna()
df.columns

# Converting into binary
lb=LabelEncoder()
df1["class_variable"]=lb.fit_transform(df1["class_variable"])

df1["class_variable"].unique()
df1['class_variable'].value_counts()
colnames=list(df1.columns)

predictors=colnames[:8]
target=colnames[8]

# Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.3)

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



