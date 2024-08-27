# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:26:43 2023

@author: Admin
"""
########## Text Mining ############
sentence="we are learning TextMining from Sanjivani AI"
###if we want know position of learning
sentence.index("learning")
###It will show learning is at position 7
### This is going to show character position from 0 in cludin

############# 
# we want to know position of Textmining word
sentence.split().index("TextMining")
## It will split the words in list and count the position
## if you want to see the list select sentence.split and ------
## it will show at 3

#############
#Suppose we want print any word in reverse order
sentence.split()[2][::-1] 
### [start:end : -1(start)] will start from -1 ,-2,-3 till the end
# learning will be printed as gninrael

############
## suppose want to print first and last word of the sentence
words=sentence.split()
first_word=words[0]
first_word
last_word=words[-1]
last_word
#now we want to concat the first and last word
concat_word=first_word+" "+last_word
concat_word

###################
# we want to print even words from the sentences
[words[i] for i in range(len(words)) if i%2==0]
# words having odd length will not be printed

############
sentence
#now we want to display only AI 
sentence[-3:]

##############
# suppose we want to display entire sentence in reverse order
sentence[::-1]
# IA inavijnaS morf gniniMtxeT gninrael era ew

################
# suppose we want to select each word and print in reversed order
words
print(" ".join(word[::-1] for word in words))
##ew era gninrael gniniMtxeT morf inavijnaS IA

########### Tokenization
import nltk
nltk.download('punkt')
from nltk import word_tokenize
words=word_tokenize("I am reading NLP Fundamentals")
print(words)

######################
# parts of speech (PoS) tagging
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(words)
## it  is going mention parts of speech

########################
#stop words from NLTK library
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words=stopwords.words('English')
## you can verify 179 stop words in variable explorer
print(stop_words)
sentence1="I am learning NLP:It is one of the most popular library in python"
#first we will tokenize the sentence
sentence_words=word_tokenize(sentence1)
print(sentence_words)
## now let us filter the  sentence using stop_words
sentence_no_stops=" ".join([words for words in sentence_words if words not in stop_words])
print(sentence_no_stops)
sentence1
## you can notice that am,is,of,the most ,popular,in are missing

###################
#suppose we want to replace words in string 
sentence2="I visited MY from IND on 14-02-19"
normalized_sentence=sentence2.replace('MY','Malaysia').replace('IND','India')
normalized_sentence=normalized_sentence.replace('-19','-2020')
print(normalized_sentence)

################
#suppose we want auto correction in the sentence
from autocorrect import Speller
#declare the function Speller defined for English
spell=Speller(lang='en')
spell('Engilish')

####################
#suppose we want to correct whole sentence
sentence3="Ntural lanagage processin deals withh the aart of extracting sentiiments"
###let us first tokenize this sentence
sentence3=word_tokenize(sentence3)
corrected_sentence=" ".join([spell(word) for word in sentence3])
print(corrected_sentence)

#####################
# stemming
stemmer=nltk.stem.PorterStemmer()
stemmer.stem("programming")
stemmer.stem("programed")
stemmer.stem("Jumping")
stemmer.stem("Jumped")

#################
##Lematizer
# lematizer looks into dictionary words
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")
lemmatizer.lemmatize("battling")
lemmatizer.lemmatize("amazing")

####################################
#chunking  (shallow parsing ) Identifying named entities
nltk.download("maxent_ne_chunker")
nltk.download('words')
sentence4="We are learning NLP in python by SanjivaniAI"
#first we will tokenize
words=word_tokenize(sentence4)
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]

'''
total output: [Tree('NE', [('NLP', 'NNP')]), Tree('NE', [('SanjivaniAI', 'NNP')])]
'''

############################
#sentence tokenization
from nltk.tokenize import sent_tokenize
sent=sent_tokenize("we are learning NLP in Python. Delivered by SanjivaniAI")
sent

#############################
from nltk.wsd import lesk
sentence1="keep your saving in the bank"
print(lesk(word_tokenize(sentence1),'bank'))
###
sentence2="It is so riskyto drive over the banks of river"
print(lesk(word_tokenize(sentence2),'bank'))
###Synset('bank.v.07)
##########
# Synset('bank.v.07) a slope in the turn of a road or track;
# the outside is higher than the inside in order to reduce the 
###
#bank as multiple meanings . if you want to find exact meaning 
#execute following code
#the definitions for "bank " can be 
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank'):print(ss,ss.definition())





