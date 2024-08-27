# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:50:34 2023

@author: Admin
"""
import pandas as pd 
import numpy as np

#read the csv
df=pd.read_csv("C:/Datasets/spam.csv")
#Check forst 10 records
df.head()
#Total number of spam and ham
df.Category.value_counts()
#create one more column comprises 0 and 1 
#name of column is spam 
df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
df.shape
#####
#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df.Message, df.spam, test_size=0.2)
#Let us check the shape of X_train data and X_test data
X_train.shape
X_test.shape
#let s check the type of X_train and y_train
type(X_train)
type(y_train)
############
#Create bag of words representation using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
X_train_cv=v.fit_transform(X_train.values)
X_train_cv
#After  creation of BoW , let us check the shape 
X_train_cv.shape
############
#Train the native bayes model 
from sklearn.naive_bayes import MultinomialNB
#Initilize the model
model=MultinomialNB()
#Train the model
model.fit(X_train_cv, y_train)
###########
#Create Bag of Words representation using CountVectorizer
# of X_test
X_test_cv=v.transform(X_test)
############
#Evaluate Performance
from sklearn.metrics import classification_report
y_pred=model.predict(X_test_cv)
print(classification_report(y_test,y_pred))

