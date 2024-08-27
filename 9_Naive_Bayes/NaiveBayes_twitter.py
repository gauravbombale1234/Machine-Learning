# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:16 2024

@author: Gaurav 
"""
"""
In this case study, you have been given Twitter 
data collected from an anonymous twitter handle.
 With the help of a Naïve Bayes model, predict 
if a given tweet about a real disaster is real or fake.

1.	Business Problem
1.1.	Though Social media has become almost an inevitable 
        part of our society now, 
       we can’t always depend on it. 
      The spreading of misinformation in social media is 
      not new. Every day we read bunches of things online on 
      social media, which may happen to be true, often is not. 
      This false or misinformation leads to fake news 
      i.e. consisted of fabricated stories, 
      without any verifiable facts, sources, or quotes. 

      1.1.2 Those stories are forged to influence reader’s 
        own opinions or to deceive them. 
        The question of fake news refers to the point of 
        how to think about the nature of real news.
      
1.2.	Are there any constraints?
        Given that the goal of content moderation is to
        decide what content should and shouldn’t be made
        widely available, thereby influencing what people
        do and do not believe, there must be a concerted 
        effort to ensure that what moderators flag as “fake” 
        content is actually fake. However, accurately identifying 
        fake news requires that interested parties agree on what 
        makes fake content problematic, such that it merits removal, 
        and that content moderators can reliably distinguish this content
        from non-problematic content. There are thus two sources of 
        disagreement related to detecting fake news: 
        (1) disagreement regarding what content should be 
        subject to moderation and (2) disagreement regarding whether 
        that content is categorized accurately.

"""
##2.	Work on each feature of the dataset to create a data dictionary as displayed in the below
#id:intnteger type which of no use
#keyword:object type
#location:object type
#text:which comrises of messages
#target:is a fake or genuine
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
#0.790407358738502
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
###let us check the wrong classified actual is fake and predicted is not fake is 68
#actual is not fake but predicted is fake is 575 ,this is not accepted
#################################################
##Multinomial Naive Bayes with laplace smoothing
###in order to address problem of zero probability laplace smoothing is used
classifier_mb_lap=MB(alpha=0.25)
classifier_mb_lap.fit(train_tfidf,tweet_train.target)
###Evaluation on test data
test_pred_lap=classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap=np.mean(test_pred_lap==tweet_test.target)
accuracy_test_lap
##it is 0.79106438896
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
###let us check the wrong classified actual is fake and predicted is not fake is 71
#actual is not fake but predicted is fake is 271 ,this is not accepted
#############################################################
#Write about the benefits/impact of the solution - 
#in what way does the business (client) benefit from 
#the solution provided?
#The spread of fake news on the Internet is a 
#cause of great concern for all members of society, 
#including the government, policymakers, organisations,
# businesses and citizens. 
#Fake news is specifically designed to plant a seed 
#of mistrust and exacerbate the existing social and cultural
# dynamics by misusing political, regional and religious
# undercurrents .These kind of model will help client to avoid 
#these situations.