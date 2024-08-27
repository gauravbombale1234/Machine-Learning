# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:16 2024

@author: Gaurav 
"""
"""
Problem Statement: -
This dataset contains information of users in a social network.
 This social network has several business clients which can post ads on it. 
 One of the clients has a car company which has just 
 launched a luxury SUV for a ridiculous price. 
 Build a Bernoulli Naïve Bayes model using this dataset and 
 classify which of the users of the social network are going to purchase this luxury SUV. 1 implies that there was a purchase and 0 implies there wasn’t a purchase.

1.	Business Problem
1.1.	What is the business objective?
      1.1.1This will help you bring those audiences to your website 
      who are interested in cars. 
      And, there will be many of those who are planning to buy a car in the near future.


      1.1.2 Communicating with your target audience over social media 
      is always a great way to build a good market reputation.
      Try responding to people’s automobile related queries on Twitter and Facebook 
      pages quickly to be their first choice when it comes to buying a car.
1.2.	Are there any constraints?
        Not having a clear marketing or social media strategy may result in reduced benefits for your business

         Additional resources may be needed to manage your online presence

        Social media is immediate and needs daily monitoring

        If you don't actively manage your social media presence, you may not see any real benefits

        There is a risk of unwanted or inappropriate behavior on your site including bullying and harassment

        Greater exposure online has the potential to attract risks. Risks can include negative feedback information, leaks, or hacking


"""
#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below
#User ID:integer type which is not contributory
#Gender:Object Type need to be label encoding
#Age:Integer
#EstimatedSalary:Integer
#Purchased:Integer Type
################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Let us first import the data set
car=pd.read_csv("C:/8_Naive_Bayes/NB_Car_Ad.csv")

#Exploratory data analysis
car.columns
car.dtypes
car.describe()
#min age of employee is 18 yeras
#max age of emploee is 60 years
#avarge age is 37.65
#min salary of user is 15000
# max salary is 1,50,000
#average salary is 69742
car.isna().sum()
car.drop(['User ID'],axis=1,inplace=True)
car.dtypes
plt.hist(car.Age)
#Age is normally distributed
plt.hist(car.EstimatedSalary)
#Data is normally distributed but right skewed
###############################################

#.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc.
car.dtypes

#The column gender is of object type
#Let us apply label encoder to input features

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
#This is model of label_encoder which is applied to all the object type columns
car['Gender']=label_encoder.fit_transform(car['Gender'])
 
#Now let us apply normalization function
def norm_funct(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
car_norm=norm_funct(car)       
car_norm.describe()       
##################################################
#Now let us designate train data and Test data

from sklearn.model_selection import train_test_split
car_train,car_test=train_test_split(car_norm,test_size=0.2)

col_names1=list(car_train.columns)
train_X=car_train[col_names1[0:2]]
train_y=car_train[col_names1[3]]
col_names2=list(car_train.columns)
test_X=car_test[col_names2[0:2]]
test_y=car_test[col_names2[3]]

############################################
##	Model Building
#Build the model on the scaled data (try multiple options).
#Build a Naïve Bayes model.
#Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, 
#BernoulliNB is designed for binary/boolean features.
from sklearn.naive_bayes import BernoulliNB as BB
classifier_bb=BB()

classifier_bb.fit(train_X,train_y)
#Let us now evaluate on test data
test_pred_b=classifier_bb.predict(test_X)
##Accuracy of the prediction
accuracy_test_b=np.mean(test_pred_b==test_y)
accuracy_test_b
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_b,test_y)
pd.crosstab(test_pred_b,test_y)
################################################
#Let us now evaluate on train data
train_pred_b=classifier_bb.predict(train_X)
##Accuracy of the prediction
accuracy_train_b=np.mean(train_pred_b==train_y)
accuracy_train_b
#0.0.64375
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(train_pred_b,train_y)
pd.crosstab(train_pred_b,train_y)
################################################
##Multinomial Naive Bayes with laplace smoothing
###in order to address problem of zero probability laplace smoothing is used
classifier_bb_lap=BB(alpha=0.25)
classifier_bb_lap.fit(train_X,train_y)

#Let us now evaluate on test data
test_pred_lap=classifier_bb_lap.predict(test_X)
##Accuracy of the prediction
accuracy_test_lap=np.mean(test_pred_lap==test_y)
accuracy_test_lap
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap,test_y)
pd.crosstab(test_pred_lap,test_y)
############################################
#Let us now evaluate on train data
train_pred_lap=classifier_bb.predict(train_X)
##Accuracy of the prediction
accuracy_train_lap=np.mean(train_pred_lap==train_y)
accuracy_train_lap
#0.7729
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(train_pred_lap,train_y)
pd.crosstab(train_pred_lap,train_y)
########################################################
# Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#The auto industry is synonymous with loyalty. 
#But as the automobile business landscape becomes even more competitive, 
#savvy dealers are doing all they can to foster long-term relationships with customers through social media.
# Used in conjunction with email campaigns and a strong web presence, 
#auto dealers who use social media can boost sales not only of vehicles 
#but also of parts and services to repeat, happy customers.