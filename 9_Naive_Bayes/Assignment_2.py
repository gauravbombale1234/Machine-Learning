# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:16 2024

@author: Gaurav 
"""

'''
1. Business Problem
1.1 Business Objective:
The business objective is to predict which users are likely to purchase the 
luxury SUV based on their information in the social network. This prediction can 
be used by the car company to target potential customers with advertisements.

1.2 Constraints:

The data provided should be representative of the target population.
The model should have a reasonably high accuracy to be useful for the business.

2. Data Dictionary
Feature	          Data Type	    Description	                              Relevant to Model Building
-------------------------------------------------------------------------------------------------------
User ID	          Numeric	    Unique identifier for each user	          No
Gender	          Categorical	Gender of the user (e.g., Male, Female)	  Yes
Age	              Numeric	    Age of the user	                          Yes
EstimatedSalary	  Numeric	    Estimated salary of the user	          Yes
Purchased	      Binary	    Indicates whether the user purchased      Yes
                                the luxury SUV (0 or 1)

'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Loading the social network dataset
social_data = pd.read_csv("C:/8_Naive_Bayes/NB_Car_Ad.csv")

# Cleaning the data (if necessary)
# You may need to preprocess the 'Gender' column if it's categorical, and handle missing values.

# Selecting relevant features (User ID, Gender, Age, EstimatedSalary) and target variable (Purchased)
features = social_data[['Gender', 'Age', 'EstimatedSalary']]
target = social_data['Purchased']

# Splitting the data into training and testing sets
social_train, social_test = train_test_split(social_data, test_size=0.2, random_state=42)

# Creating matrix of token counts for the entire feature set
def split_into_words(text):
    return [word for word in str(text).split()]

social_bow = CountVectorizer(analyzer=split_into_words).fit(features.apply(lambda x: ' '.join(x.astype(str)), axis=1))
all_social_matrix = social_bow.transform(features.apply(lambda x: ' '.join(x.astype(str)), axis=1))

# Transforming messages for training
train_social_matrix = social_bow.transform(social_train.apply(lambda x: ' '.join(x[['Gender', 'Age', 'EstimatedSalary']].astype(str)), axis=1))

# Transforming messages for testing
test_social_matrix = social_bow.transform(social_test.apply(lambda x: ' '.join(x[['Gender', 'Age', 'EstimatedSalary']].astype(str)), axis=1))

# Learning term weighting and normalization on the entire dataset
tfidf_transformer = TfidfTransformer().fit(all_social_matrix)

# Preparing TF-IDF for train data
train_tfidf = tfidf_transformer.transform(train_social_matrix)

# Preparing TF-IDF for test data
test_tfidf = tfidf_transformer.transform(test_social_matrix)

# Building Naïve Bayes model (using Multinomial Naïve Bayes for non-negative features)
classifier_nb = MultinomialNB()
classifier_nb.fit(train_tfidf, social_train['Purchased'])

# Predicting on test data
test_pred_nb = classifier_nb.predict(test_tfidf)

# Evaluating the model
accuracy_test_nb = accuracy_score(test_pred_nb, social_test['Purchased'])
print(f"Accuracy on test data: {accuracy_test_nb}")


##############################################################
###############################################################

######### 2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Let us first import the data set
salary_train=pd.read_csv("C:/8_Naive_Bayes/SalaryData_Train.csv")

# 2nd data set
salary_test=pd.read_csv("C:/8_Naive_Bayes/SalaryData_Test.csv")
#Exploratory data analysis
salary_train.columns
salary_train.dtypes
salary_train.describe()
#min age of employee is 17 yeras
#max age of emploee is 90 years
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


##########################################
################  3 
'''
1 Business Objective:

The primary business objective is to develop a Naïve Bayes model that 
can accurately predict whether a given tweet about a disaster is real (1) 
or fake (0). The model aims to assist in identifying genuine 
disaster-related information on Twitter, aiding in timely responses 
and crisis management.
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import re
#Now let us load the data
tweet=pd.read_csv("C:/8_Naive_Bayes/Disaster_tweets_NB.csv")
#Exploratory data analysis
tweet.columns
tweet.dtypes
##Since there are no numeric data hence further EDA is not possible
#let us conver the text messages to TFIDF 
#Let us clean the data
def cleaning_text(i):
    i=re.sub("[^A-Za-z""]+"," ",i).lower()
    #Let us declare empty list
    w=[]
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return("  ".join(w))
#Let us check the function
cleaning_text("This is just trial text to check the cleaning_text method")
cleaning_text("Hi how are you ,I am sad")
#There could messages which will create empty spaces after the cleaning
#Now first let us apply to the tweet.text column
tweet.text=tweet.text.apply(cleaning_text)
#The numer rows of tweet is 7613
#Let us first drop the columns which are not useful
tweet.drop(["id","keyword","location"],axis=1,inplace=True)
#There could messages which will create empty spaces after the cleaning
tweet=tweet.loc[tweet.text !="",:]
#The number of rows are reduced to 7610 after this cleaning
###########################################
###let us first split the data in training set and Testing set
from sklearn.model_selection import train_test_split
tweet_train,tweet_test=train_test_split(tweet,test_size=0.2)
#####################################
#let us first tokenize the message
def split_into_words(i):
    return[word for word in i.split(" ")]
#This is tokenization or custom function will be used for CountVectorizer
tweet_bow=CountVectorizer(analyzer=split_into_words).fit(tweet.text)
#This is model which will be used for creating count vectors
#let us first apply to whole data
all_tweet_matrix=tweet_bow.transform(tweet.text)
#Now let us apply to training messages
train_tweet_matrix=tweet_bow.transform(tweet_train.text)
#similarly ,let us apply to test_tweet
test_tweet_matrix=tweet_bow.transform(tweet_test.text)
##Let us now apply to TFIDF Transformer
tfidf_transformer=TfidfTransformer().fit(all_tweet_matrix)
##This is being used as model.let us apply to train_tweet_matrix
train_tfidf=tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape
#let us now apply it to test_tweet_matrix
test_tfidf=tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape
##################################################
#let us now apply it to Naive model
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
#let us train the model
classifier_mb.fit(train_tfidf,tweet_train.target)
#############################
# let us now evaluate the model with test data
test_pred_m=classifier_mb.predict(test_tfidf)
##Accuracy of the prediction
accuracy_test_m=np.mean(test_pred_m==tweet_test.target)
accuracy_test_m
#To find the confusion matrix
from sklearn.metrics import accuracy_score
pd.crosstab(test_pred_m,tweet_test.target)
###let us check the wrong classified actual is fake and predicted is not fake is 64
#actual is not fake but predicted is fake is 255 ,this is not accepted
######################################
#Let us evaluate the model with train data
train_pred_m=classifier_mb.predict(train_tfidf)
accuracy_train_m=np.mean(train_pred_m==tweet_train.target)
accuracy_train_m
###let us check the confusion matrix
pd.crosstab(train_pred_m,tweet_train.target)
classifier_mb_lap=MB(alpha=0.25)
classifier_mb_lap.fit(train_tfidf,tweet_train.target)
###Evaluation on test data
test_pred_lap=classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap=np.mean(test_pred_lap==tweet_test.target)
accuracy_test_lap
from sklearn.metrics import accuracy_score
pd.crosstab(test_pred_lap,tweet_test.target)
###let us check the wrong classified actual is fake and predicted is not fake is 103
#actual is not fake but predicted is fake is 215 ,this is not accepted
######################
#Training data accuracy
train_pred_lap=classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap=np.mean(train_pred_lap==tweet_train.target)
accuracy_train_lap
pd.crosstab(train_pred_lap,tweet_train.target)