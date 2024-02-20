# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:12:08 2023

@author: Gaurav Bombale
"""
import pandas as pd
import matplotlib.pyplot as plt

# now import file from dataset and create a dataframe
Univ1=pd.read_excel("C:/Datasets/University_Clustering.xlsx")
a=Univ1.describe()

Univ= Univ1.drop(['State'],axis=1)
# we know that there is scale difference among the columns, which we have
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



#############################################################
################################################################
#### For Autoinsurance.csv file(remaining)
import pandas as pd
import matplotlib.pyplot as plt

# now import file from dataset and create a dataframe
Univ1=pd.read_csv("C:/Datasets/AutoInsurance.csv")
a=Univ1.describe()

Univ= Univ1.drop(['Customer','Effective To Date','State','Response','Coverage','Education','EmploymentStatus','Gender','Location Code','Marital Status','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class','Vehicle Size'],axis=1)
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

