# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:03:39 2023

@author: Gaurav Bombale

Assignments :
"""
'''
1.	Perform clustering for the airlines data to obtain optimum number of 
clusters. Draw the inferences from the clusters obtained. Refer to 
EastWestAirlines.xlsx dataset.
'''
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("C:/Datasets/EastWestAirlines.xlsx")
df.head()
df.columns
# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization
df=df.drop(['ID#'],axis=1)

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df.iloc[:,:])

b=df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")

# sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
# dendrongram()
# applying agglomerative clustering choosing 3 as clustrers
# from dendrongram
# whatever has been displayed in dendrogram is not clustering
# It is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

# Assign this series to df Dataframe as column and name the column
df['clust']=cluster_labels
# we want to restore the column 7 to 0 th position
df.shape
df=df.iloc[:,[11,1,2,3,4,5,6,7,8,9,10]]
# now check the df dataframe
df.iloc[:,2:].groupby(df.clust).mean()
# from the output cluster 2 has got highest Top10
# lowest accept ratio , best faculty ratio and highest expenses
# highest graduate 

df.to_csv("EastWestAirlinesResult.csv",encoding="utf-8")
import os
os.getcwd()





'''
2.	Perform clustering for the crime data and identify the number 
of clusters formed and draw inferences. 
Refer to crime_data.csv dataset.
'''
import pandas as pd
import matplotlib.pyplot as plt

# now import file from dataset and create a dataframe
Univ1=pd.read_csv("C:/Datasets/crime_data.csv")
a=Univ1.describe()
a
Univ= Univ1.drop(['Unnamed: 0'],axis=1)
# we know that there is scale difference among the columns, whuch we have
# either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# now apply this normalization function to Univ dataframe for all the 
# rows and column from 1 until end
# since 0th column has University name hence skipped
df_norm=norm_func(Univ.iloc[:,:])
#you can check the df_norm dataframe which is scaled 
#between values from 0 to 1
#you can apply describe function to the new dataframe
b=df_norm.describe()
#before you apply clustering , you need to plotdendogram first
# Now to create dendogram , we need to measure the distance
# we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering 
# ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendogram ")
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendogram()
#applying aglomerative clustering choosing 3 as clusters
# from dendogram 
# whatever has been  displayed in dendogram is not clustering 
# it is just showing  number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)
# apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to unix Dataframe as column and name the column
Univ['clust']=cluster_labels
#we wnat to relocate the column 7 to 0th position
Univ.shape
Univ1=Univ.iloc[:,[0,1,2,3,4]]
#now check the Univ1 dataframe
Univ1.iloc[:,:].groupby(Univ1.clust).mean()
#from the output cluster 2 has got highest Top10
#lowest accept ratio , best faculty ratio and highest expenses
# highest graduates ratio
Univ1.to_csv("CrimeDataAssignResult.csv",encoding="utf-8")
import os 
os.getcwd()



'''
3.	Perform clustering analysis on the telecom data set. 
The data is a mixture of both categorical and numerical data. 
It consists of the number of customers who churn out. 
Derive insights and get possible information on factors that may 
affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.
'''
import pandas as pd
import matplotlib.pyplot as plt

# now import file from dataset and create a dataframe
Univ1=pd.read_excel("C:/Datasets/Telco_customer_churn.xlsx")
a=Univ1.describe()
a
Univ= Univ1.drop(['Customer ID','Count','Quarter','Referred a Friend','Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract','Paperless Billing','Payment Method'],axis=1)
# we know that there is scale difference among the columns, whuch we have
# either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# now apply this normalization function to Univ dataframe for all the 
# rows and column from 1 until end
# since 0th column has University name hence skipped
df_norm=norm_func(Univ.iloc[:,1:])
#you can check the df_norm dataframe which is scaled 
#between values from 0 to 1
#you can apply describe function to the new dataframe
b=df_norm.describe()
#before you apply clustering , you need to plotdendogram first
# Now to create dendogram , we need to measure the distance
# we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering 
# ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendogram ")
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendogram()
#applying aglomerative clustering choosing 3 as clusters
# from dendogram 
# whatever has been  displayed in dendogram is not clustering 
# it is just showing  number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)
# apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to unix Dataframe as column and name the column
Univ['clust']=cluster_labels
#we wnat to relocate the column 7 to 0th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
#from the output cluster 2 has got highest Top10
#lowest accept ratio , best faculty ratio and highest expenses
# highest graduates ratio
Univ1.to_csv("University.csv",encoding="utf-8")
import os 
os.getcwd()


