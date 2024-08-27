# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:26:04 2023

@author: Gaurav Bombale
"""
#pip install mlxtend

from mlxtend.frequent_patterns import apriori,association_rules
#here we are going to use transactional data wherein size of each row is not consistent
#we can not use pandas to load this unstructured data
#here function called open() is used
# create an empty list
groceries=[]
with open("‪C:/Datasets/groceries.csv") as f:groceries=f.read()
#spliting the data into separate transactions using separator , it is coma
# we can use new line character was ("\n")
groceries=groceries.split("\n")
#Earlier groceries datastruture was in string format , now it will change to
#9836 , each item is coma separated
#we will have to separate out each item from each transaction
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
#split function will seperate each item from each list, wherever it will find 
#in order to generate association rules, you can directly use groceries_list 
#Now let us seperate out each item from the groceries_list 
all_groceries_list=[i for item in groceries_list for i in item]
#You will get all the items occured in all transactions 
#we will get 43368 items in various transactions

# Now let us count the frequency of each item
#we will import collections package which has Counter function which will
from collections import Counter
item_frequencies=Counter(all_groceries_list)
#item_frequencies is basically having x[0] as key and x[1] =values
#we want to access values and sort based on the count that occured in it
#it will show the count of each item purchased in every transaction
#Now let us sort these frequencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1]) 
# whe we will execute this , item frequencies will be in sorted form 
#in this form of tuple 
#item name with count
#Let us separate out items and their count
items=list(reversed([i[0] for i in item_frequencies]))
#this is list comprehension for each item in item frequencies access the key
#there you will get item list
frequencies=list(reversed([i[1] for i in item_frequencies]))
#where you will get count of purchase of each item

#now let us plot bar graph of item frequencies
import matplotlib.pyplot as plt
#here we are taking frequencies from zero to 11 , you can try 0-15 or any other
plt.bar(height=frequencies[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])
#plt.xticks , you can specify a rotation for the tick
#labels in degrees or with keyword
plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#now let us try to establish association rule mining
#we have grocerries list in the list format , we need 
# to convert it in dataframe format
groceries_series=pd.DataFrame(pd.Series(groceries_list))
#now we will get dataframe of size 9836x1 size,column
#comprises of multiple items
#we had extra row created, check the groceries_series,
#last row is empty , let us first delete it
groceries_series=groceries_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and columns 0 to all
#groceries series has column having name 0 , let us rename as transactions
groceries_series.columns=["Transactions"]
#Now we will have to apply 1-hot encoding, before that in one column there are various items seperated by ','
#let us seperate it with '*'
x=groceries_series["Transactions"].str.join(sep='*')
#check the x in variable explorer which has * seperator rather that ','
x=x.str.get_dummies(sep='*')
#you will get one hot encoded data frame of size 9835x169
#This is our input data to applyy to apriori algorithm, it will generate !169 rules, min support values 
#is 0.0075 (it must be between 0 to 1),
#you can give any number but must be between 0 to 1 
frequent_itemsets = apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
#you will get support  values for 1,2,3 and 4 max items 
#let us sort this support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order
#Even EDA was also have the same trend , in EDA there was count
# and here it is support values
# we will generate associatian rules , This association 
# rule will calculate all the matrix
# of each and every combination 
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
#this generate associatin rules of size 1198x9 columns 
#comprises of antescends, consequences 
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

