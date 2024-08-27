# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:02:59 2023

@author: Admin
"""
############## Tokenization  #######################
import re
sentence5="sharat twitted ,Wittnessing 70th republic day India from Rajpath,\new Delhi, Mesmorizing performance by Indian Army|"
re.sub(r'([^\s\w]|_)+',' ', sentence5).split()
# extracting n-grams
# n-gram can be extracted using three techniques
# 1.custom defined function
# 2.nltk
# 3. TextBlob
#####################
# extracting n-grams using custom defined function
import re
def n_gram_extracter(input_str,n):
    tokens=re.sub(r'([^\s\w]|_)+',' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])

n_gram_extracter("The cute little boy is playing with kitten",2)
n_gram_extracter("The cute little boy is playing with kitten",3)

###############
from nltk import ngrams 
#extraction n-grams with nltk
list(ngrams("The cute little boy is playing with kitten".split(),2))
list(ngrams("The cute little boy is playing with kitten".split(),3))

###############
from textblob import TextBlob
blob=TextBlob("The cute little boy is playing with kitten")
blob.ngrams(n=2)
blob.ngrams(n=3)

#################
### Tokenization using Keras
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)

###################
#Tokenization using textblob
from textblob import TextBlob
blob=TextBlob(sentence5)
blob.words

###############
## Tweet Tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)

############
### Multi_Word_Expression
from nltk.tokenize import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace("!"," ").split())

##############
### Regular Expression Tokenizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\s+')
reg_tokenizer.tokenize(sentence5)

##############
## White space tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer=WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)

#################
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer=WordPunctTokenizer()
wp_tokenizer.tokenize(sentence5)

##################### stemming(remove 'ing' word)
sentence6="I love playing cricket.Cricket players practices hard in their inning"
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd) for wd in sentence6.split())

#################
sentence7='Before eating ,it would e nice to  sanitize your hand with a sanitizer'
from nltk.stem.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])

#################
### Lemmatization 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
sentence8='The codes executed today are far better than what we execute generally'
words=word_tokenize(sentence8)
" ".join([lemmatizer.lemmatize(word) for word in words])
'''ouput : The code executed today are far better than what we execute generally'''

#####################
### singularize and pluralization 
from textblob import TextBlob
sentence9=TextBlob("She sells seashells in the seashore")
words=sentence9.words
##we want to make word[2] i.e. seashell in singular form 
sentence9.words[2].singularize()
##we want words 5 i.e. seashore in plural form
sentence9.words[5].pluralize()

################
#  language translation from spanish to English
from textblob import TextBlob
en_blob=TextBlob(u'muy bien')
en_blob.translate(from_lang='es',to='en')
# es:spanish  en:english

##################
##  custom stopwords removal
from nltk import word_tokenize
sentence9='She sells seashells on the seashore'
custom_stop_word_list=['she','on','the','am','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower() not in custom_stop_word_list])
# select words which are not in defined list

############################
#extracting general features from raw text
#number of words
#detect presence of wh word
#polarity 
#subjectivity
#language identification

#########
#To identify the number of words
import pandas as pd
df=pd.DataFrame([['The vacine for covid-19 will be anounced on 1 st August'],['Do you know how much expections the world population is having from this research?'],['The risk of virus will come to an end on 31st Jul']])
df.columns=['text']
df
#now let us measure the number of words
from textblob import TextBlob
df['number_of_words']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']

###############################
#Detect presence of words wh
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

#######################
#Polarity of the sentence
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
sentence10='I like this example very much'
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10='This is fantastic example and I like it very much'
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10='This is helpful example but I would have prefer another one'
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10='This is my personal opinoin that it was helpful example but I would prefer another one'
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10='I do not like , It is bad'
pol=TextBlob(sentence10).sentiment.polarity
pol

#######
#Subjectivity of the dataframe df and check whether there is 
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

#####
# To find language of the sentence , this part of code will get http error
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())

############
# Bag of Words
# This BoW converts unstructured data to the structured form
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus=["At least seven idian pharma companies are working to develop vaccine against the corona virus.","The deadly virus that has already infected more than 14 million globally","Bharat Biotech is the among the domastic pharma firm working on the corona virus vaccine in India"]
bag_of_word_model=CountVectorizer()
print(bag_of_word_model.fit_transform(corpus).todense())
bag_of_word_df=pd.DataFrame(bag_of_word_model.fit_transform(corpus).todense())
#This will create DataFrame
bag_of_word_df.columns=sorted(bag_of_word_model.vocabulary_)
bag_of_word_df.head()


#############################
# bag of words model small
bag_of_word_model_small=CountVectorizer(max_features=5)
bag_of_word_df_small=pd.DataFrame(bag_of_word_model_small.fit_transform(corpus).todense())
bag_of_word_df_small.columns=sorted(bag_of_word_model_small.vocabulary_)
bag_of_word_df.head()

##############
##How to use TFIDF
import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
corpus=['The mouse had a tiny little mouse','The cat saw the mouse','The cat catch the mouse','The end of mouse story']
#step 1 initialize count vector 
cv=CountVectorizer()
#To count the total no. of TF
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
#Now next step is to apply IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
#This matrix is in raw matrix form , let us convert it in dataframe
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=['idf_weights'])
#sort ascending
df_idf.sort_values(by=['idf_weights'])















