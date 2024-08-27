# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:45:05 2024

@author: Admin
"""
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)

df=pd.DataFrame(iris.data , columns=iris.feature_names)
df.head()

df['target']=iris.target
df.head()

X=df.drop('target',axis='columns')
y=df.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test,y_test)

#let us change the parameters
model=RandomForestClassifier(n_estimators=40)
#n_estimators: number of trees in the forest
model.fit(X_train, y_train)

model.score(X_test,y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm
#%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')