# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:16 2024

@author: Gaurav 
"""
"""
Prepare a classification model using the Naive Bayes algorithm 
for the salary dataset. Train and test datasets are given separately. 
Use both for model building. 

1.	Business Problem
1.1.	What is the business objective?
      1.1.1The motivation of human objectives in a business
      is to find ways to meet the needs of your employees, 
      so that they feel valued and supported. 
      1.1.2 Organic business objectives are goals that incorporate all aspects of the business: 
          its development, survival, progress and outlook.
1.2.	Are there any constraints?
        information associated with each job post as text and then they represent each sample as a vector of word/keyword frequencies.
        As a consequence, these vectors are often characterized by
        a very high number of dimensions (in the order of thousands). 
        Therefore, it is necessary to collect huge amounts of job posts to be able to train a classifier or a regression model effectively.


"""
#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below
#Age:Numeric variable
#Workclass.Non numeric ,categorical
#Education.:Non numeric
#Education Number of Years.:Numeric
#Marital-status.:Non numeric
#Occupation.Non numeric
#Relationship.:Non numeric
#Race.:Non numeric
#Sex.:categorical
#Capital-gain.Numeric
#Capital-loss.:Numeric
#Hours-per-week.:Numeric
#Native-country.:Non numeric
################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Let us first import the data set
salary_train=pd.read_csv("C:/8_Naive_Bayes/SalaryData_Train.csv")
salary_test=pd.read_csv("C:/8_Naive_Bayes/SalaryData_Test.csv")
#Exploratory data analysis
salary_train.columns
salary_train.dtypes
salary_train.describe()
#min age of employee is 17 yeras
#max age of emploee is 90 years
#avarge age is 38.43
#min hours per week of the employee is 1 hour
# max hours per week of the employee is 99 hour
#average hours per week of the employee is 40.93 hours
salary_train.isna().sum()
salary_test.isna().sum()
plt.hist(salary_train.age)
#Age is right skewed
plt.hist(salary_train.educationno)
#Data is normally distributed but left skewed
###############################################

#.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc.
salary_train.dtypes
salary_test.dtypes
#Following are the columns of object type
#Let us apply label encoder to input features
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
#This is model of label_encoder which is applied to all the object type columns
for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

#Now let us apply normalization function
    def norm_funct(i):
        x=(i-i.min())/(i.max()-i.min())
        return x
salary_train_norm=norm_funct(salary_train)       
salary_test_norm=norm_funct(salary_test)        
##################################################
#Now let us designate train data and Test data
salary_train_col=list(salary_train.columns)
train_X=salary_train[salary_train_col[0:13]]
train_y=salary_train[salary_train_col[13]]

salary_test_col=list(salary_test.columns)
test_X=salary_test[salary_test_col[0:13]]
test_y=salary_test[salary_test_col[13]]
############################################
##	Model Building
#Build the model on the scaled data (try multiple options).
#Build a Naïve Bayes model.
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()

classifier_mb.fit(train_X,train_y)
#Let us now evaluate on test data
test_pred_m=classifier_mb.predict(test_X)
##Accuracy of the prediction
accuracy_test_m=np.mean(test_pred_m==test_y)
accuracy_test_m
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m,test_y)
pd.crosstab(test_pred_m,test_y)
###let us check the wrong classified actual is grater than 50 and predicted is less than 50 is 469
#actual salary prediction less than 50 but predicted is 'greater than 50' is 2920 ,this is not accepted
################################################
#Let us now evaluate on train data
train_pred_m=classifier_mb.predict(train_X)
##Accuracy of the prediction
accuracy_train_m=np.mean(train_pred_m==train_y)
accuracy_train_m
#0.7729
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(train_pred_m,train_y)
pd.crosstab(train_pred_m,train_y)
###let us check the wrong classified actual is 'grater than 50' and predicted is 'less than 50' is 936
#actual salary prediction less than 50 but predicted is 'greater than 50' is 5913 ,this is not accepted
################################################
##Multinomial Naive Bayes with laplace smoothing
###in order to address problem of zero probability laplace smoothing is used
classifier_mb_lap=MB(alpha=0.75)
classifier_mb_lap.fit(train_X,train_y)

#Let us now evaluate on test data
test_pred_lap=classifier_mb_lap.predict(test_X)
##Accuracy of the prediction
accuracy_test_lap=np.mean(test_pred_lap==test_y)
accuracy_test_lap
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap,test_y)
pd.crosstab(test_pred_lap,test_y)
###let us check the wrong classified actual is 'grater than 50' and predicted is 'less than 50 is 469
#actual salary prediction less than 50 but predicted is 'greater than 50' is 2920 ,this is not accepted
############################################
#Let us now evaluate on train data
train_pred_lap=classifier_mb.predict(train_X)
##Accuracy of the prediction
accuracy_train_lap=np.mean(train_pred_m==train_y)
accuracy_train_m
#0.7729
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(train_pred_lap,train_y)
pd.crosstab(train_pred_lap,train_y)
###let us check the wrong classified actual is 'grater than 50' and predicted is 'less than 50' is 936
#actual salary prediction less than 50 but predicted is 'greater than 50' is 5913 ,this is not accepted
########################################################
# Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#There are two class values ‘>50K‘ and ‘<=50K‘, meaning it is a binary classification task. 
#The classes are imbalanced, with a skew toward the ‘<=50K‘ class label.
#‘>50K’: majority class, approximately 25%.
#‘<=50K’: minority class, approximately 75%.
