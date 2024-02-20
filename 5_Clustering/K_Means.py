# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:42:33 2023

@author: Gaurav Bombale
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

#let us try to understand first how k means works for two 
#dimensional data
#for that , generate random numbers in the range 0 to 1
#and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=['X','Y'])
#assign the values of X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x='X',y='Y',kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)

'''
with data X and Y , apply kmeans model ,
generate scatter plot 
with scale / font = 10

cmap=plt.cm.coolwarm:cool  color combination
''' 
model1.labels_
df_xy.plot(x='X',y='Y',c=model1.labels_,kind='scatter',s=10,cmap=plt.cm.coolwarm)


Univ1=pd.read_excel("C:/Datasets/University_Clustering.xlsx")
a=Univ1.describe()

Univ= Univ1.drop(['State'],axis=1)
# we know that there is scale difference among the columns, whuch we have
# either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# now apply this normalization function to Univ dataframe for all the row
df_norm=norm_func(Univ.iloc[:,1:])
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
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("kmeans_University.csv",encoding="utf-8")
import os
os.getcwd()

