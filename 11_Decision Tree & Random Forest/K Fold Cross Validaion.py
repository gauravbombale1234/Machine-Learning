# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:30:50 2024

@author: Admin
"""
import pandas as pd

df=pd.read_csv("puma_diabetes.csv")

df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()
#0 500
#1 268
#there is slight imbalance in our dataset but since it is not 
#major we will not worry about it 

#train test split
X=df.drop('Outcome',axis='columns')
y=df.Outcome
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled[:3]
#In order to make your data balanced while splitting , you can use stratify
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,stratify=y,random_state=10)

X_train.shape
X_test.shape
y_train.value_counts()
'''
Outcome
0    375
1    201
'''
201/375
#0.536
y_test.value_counts()
'''
Outcome
0    125
1     67
'''
67/124
#Train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#Here k fold cross validation is used 
scores=cross_val_score(DecisionTreeClassifier(),X,y,cv=5)
scores
scores.mean()
#Accuracy is 0.7045072574484339

#Train using Bagging 
from sklearn.ensemble import BaggingClassifier

bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
    )


