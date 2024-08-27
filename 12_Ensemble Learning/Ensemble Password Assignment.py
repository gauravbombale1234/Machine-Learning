# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:04:52 2024

@author: Gaurav Bombale
"""
""" Problem Statement:
3.	4.	Data privacy is always an important factor to safeguard 
their customers' details. For this, password strength is an 
important metric to track. Build an ensemble model to classify 
the user’s password strength.


1.	Business Problem
1.1.	What is the business objective?
  1.1.1 Passwords are undoubtedly essential to security, but they are not
  the only method that can or should be used to protect one's computers and 
  devices. In addition to creating a good password, people should learn
  how to safeguard it and use it wisely.This means never sharing it and, 
  if unable to remember it, keeping the written copy in a secure location.
  
1.1.	Are there any constraints?

"""
#characters              object
#characters_strength    float64
#length                   int64
#capital                  int64
#small                    int64
#special                  int64
####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel("C:/11_Ensemble Learning/Ensemble_Password_Strength.xlsx")
###EDA
df.dtypes
#let us check the outliers
plt.boxplot(df.characters_strength)
#There are outliers
df.isnull().sum()
#There are  null values in Bean_type and Origin
##########################################
###Data preprocessing
df.dtypes
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["characters_strength"])
df_t=winsor.fit_transform(df[["characters_strength"]])
sns.boxplot(df_t.characters_strength)
#let us calculates the length of a given password.
length=lambda x:cal_len(x)
capital=lambda x:cal_capL(x)
small=lambda x:cal_smL(x)
special=lambda x:cal_spc(x)
def cal_len(x):
    x=str(x)
    return len(x)
#Calculates the number of capital letters in the password.
def cal_capL(x):
    x=str(x)
    cnt=0
    for i in x:
        if(i.isupper()):
            cnt+=1
    return cnt
#    Calculates the nu,ber of small letters in the password.
def cal_smL(x):
    x=str(x)
    cnt=0
    for i in x:
        if(i.islower()):
            cnt+=1
    return cnt
#Calculates the number of special characters in the password.
import re
def cal_spc(x):
     x=str(x)
     a=(len(x)-len(re.findall('[\w]',x)))
     return a

length=lambda x:cal_len(x)
capital=lambda x:cal_capL(x)
small=lambda x:cal_smL(x)
special=lambda x:cal_spc(x)

df['length']=pd.DataFrame(df.characters.apply(length))
df['capital']=pd.DataFrame(df.characters.apply(capital))
df['small']=pd.DataFrame(df.characters.apply(small))
df['special']=pd.DataFrame(df.characters.apply(special))
##############################
################
df1=df.loc[:,df.columns!='characters_strength']
predictors=df1.loc[:,df1.columns!='characters']

target=df["characters_strength"]
#################dia
#splitting dataset into train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)
##################


########################################
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model,svm,neighbors,naive_bayes
learner_1=neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2=linear_model.Perceptron(tol=1e-2,random_state=0)
learner_3 = svm.SVC(gamma=0.01)
#Now let us instantiate voting classifier
voting=VotingClassifier([('KNN',learner_1),
                          ('Prc',learner_2),
                          ('SVM',learner_3)

                        ])
#Now let us apply it to the train data set
voting.fit(x_train,y_train)
#predict the most voted class
from sklearn.metrics import accuracy_score
hard_predictions=voting.predict(x_test)
print("Hard Voting",accuracy_score(y_test,hard_predictions))
##############################
#Soft voting 
learner_4=neighbors.KNeighborsClassifier(n_neighbors=5)
learner_5 = naive_bayes.GaussianNB()
learner_6=svm.SVC(gamma=0.01,probability=True)
voting=VotingClassifier([('KNN',learner_4),
                          ('NB',learner_5),
                          ('SVM',learner_6)],
                           voting='soft'

                         )
# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))
##########################################################3
#What are the benifits to the client
#Security measures such as passwords 
#are critical when it comes to preventing the unauthorized access
# of one's computer and mobile devices. In today's world, 
# hackers and other cyber-criminals are continuously finding 
# new ways to gain access to these devices in order to steal or 
# exploit the information within. Careless use of passwords, 
# however, can be as bad as leaving one's computing devices unprotected. For this reason, people should create and protect their passwords with care.

