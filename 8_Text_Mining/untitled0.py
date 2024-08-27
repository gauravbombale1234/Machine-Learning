# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:45:32 2023

@author: Admin
"""

#pip install gensim
#pip install python-levenshtein

import gensim
import pandas as pd 
df=pd.read_json("C:/Datasets/Cell_Phones_and_Accessories_5.json",lines=True)
df
df.shape
#Simple Preprocessing & Tokenization
review_text=df.reviewText.apply(gensim.utils.simple_preprocess)
review_text
#Let us check first word of each review
review_text.loc[0]
#Let us check first row of dataframe
df.reviewText.loc[0]
#Training the Word2Vec Model 
model=gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
    )

'''
where window is how many words you are going to
consider as sliding window you can choose any count
min_count- there must min 2 words in each sentence 
workers:no. of threads

'''
#Build the Vocabulary
model.build_vocab(review_text, progress_per=1000)
#progress_per: after 1000 words it shows progress
#Train the Word2Vec model
#it will take time , have patience
model.train(review_text,total_examples=model.corpus_count, epochs=model.epochs)
#save the model
model.save("C:/7_Text_Mining/word2vec-amazon-cell-accessories-reviews-short.model")
#finding similar words and similarity between words
model.wv.most_similar('bad')
model.wv.similarity(w1='cheap',w2='inexpensive')
model.wv.similarity(w1='great',w2='good')


