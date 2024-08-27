# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:06:27 2024

@author: Gaurav

Spam message classifier by Naive Bayes Classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#loading data
email_data=pd.read_csv("C:/8_Naive_Bayes/sms_raw_NB.csv",encoding="ISO-8859-1")

#cleaning of data
import re

def cleaning_text(i):
    w = []
    i = re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


#Testing above function with some test text

cleaning_text("Hope your are having good week. just checking")
cleaning_text("hope i understand your feelings 123121.123.hi how are you")
cleaning_text("hi how are you , i am sad")
email_data.text = email_data.text.apply(cleaning_text)
email_data = email_data.loc[email_data.text != "", :]

from sklearn.model_selection import train_test_split
email_train , email_test = train_test_split(email_data , test_size=0.2)

#creating matrix of token counts for entire text documents

def split_into_words(i):
    return[word for word in i.split(" ")]

emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)
all_emails_matrix = emails_bow.transform(email_data.text)

#for transforming messages
train_emails_matrix = emails_bow.transform(email_train.text)
#for training messages
test_emails_matrix = emails_bow.transform(email_test.text)

#learning Term weightaging and normaling on entire emails

tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

#preaparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
#preaparing TFIDF for test mails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape

#now let us apply this to Naive Bayers
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_tfidf, email_train.type)
#evaluation on test data

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test.type)
accuracy_test_m
