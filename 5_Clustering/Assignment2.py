# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:58:08 2023

@author: Gaurav Bombale
"""
############### Assignment 2 #########################
'''
1.	Perform K means clustering on the airlines dataset to obtain 
optimum number of clusters. Draw the inferences from the clusters 
obtained. Refer to EastWestAirlines.xlsx dataset.
'''
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

Univ1=pd.read_excel("C:/Datasets/EastWestAirlines.xlsx")
a=Univ1.describe()
a

Univ= Univ1.drop(['ID#'],axis=1)
# we know that there is scale difference among the columns, whuch we have
# either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# now apply this normalization function to Univ dataframe for all the row
df_norm=norm_func(Univ.iloc[:,:])
'''
what will be ideal cluster number , will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)# total within sum of square
    
TWSS
# as k values increases the TWSS values decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel('No_of_clusters')
plt.ylabel('Total_within_SS')

'''
How to select value of k from elbow curve
when k changes from 2 to 3 , then decrease in 
TWSS is higher than 
when k changes from 3 to 4 .
when k values changes from 5 to 6 decrease in 
TWSS is considerably less, hence considered k=3
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[0,1,2,3,4,5,6,7]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("kmeans_EastWestAirlines.csv",encoding="utf-8")
import os
os.getcwd()


