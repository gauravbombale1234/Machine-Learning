# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:48:22 2023

@author: Gaurav Bombale
"""

"""
⦁	Perform K means clustering on the airlines dataset to obtain 
optimum number of clusters. Draw the inferences from the clusters 
obtained. Refer to EastWestAirlines.xlsx dataset.
"""

"""
Business Objective 
Minimize : charges or cost
Maximaze : Offers 

Business constraints : Customer Satisfaction 
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("C:/datasets/EastWestAirlines.xlsx")
df.head()

df.columns
# Name of columns
'''['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?']'''

# Data Dictionary
fea={
    'Name_of_feature':['ID', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
           'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
           'Days_since_enroll', 'Award'],
    'Discription':['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
           'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
           'Days_since_enroll', 'Award?'],
    'Type':['Quantitative','Quantitative','Quantitative','ordinal','ordinal','ordinal',
            'Quantitaive','Quantitative','Quantitative','Quantitative',
            'continous','ordinal'],
    'Relevance':['Irrelevance','Relevance','Relevance','Relevance','Relevance','Relevance',
            'Relevance','Relevance','Relevance','Relevance',
            'Relevance','Relevance']
    
    }

data_dict=pd.DataFrame(fea)
data_dict



""" EDA """

df.shape
# (3999, 12) - 12 columns

df.dtypes
'''
All the data integer form no need to change data types

ID#                  int64
Balance              int64
Qual_miles           int64
cc1_miles            int64
cc2_miles            int64
cc3_miles            int64
Bonus_miles          int64
Bonus_trans          int64
Flight_miles_12mo    int64
Flight_trans_12      int64
Days_since_enroll    int64
Award?               int64

'''

df.columns
'''
Index(['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?'],
      dtype='object')
'''
# Five number summary
df.describe()

# Check for null values
df.isnull()
# False

# calculating the sum of all null values 
df.isnull().sum()
# 0

# if any null then drop it
# df.dropna()

df.isnull().sum()

# mean 
df.mean()

#####################################################

import seaborn as sns
import matplotlib.pyplot as plt

# histplot

sns.histplot(df['Balance'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df['Bonus_miles'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 

######################################################

# box plot on column
sns.boxplot(df.Balance)
# There is outliers

# box plot on column
sns.boxplot(df.Bonus_miles)

# box plot on all dataframe
sns.boxplot(data=df)
# There is outliers on Balance and many columns

# Scatterplot on column
sns.scatterplot(df.Balance)

# Scatterplot on column
sns.scatterplot(df.Bonus_miles)

# Scatter plot 
# Here 'sns' corresponds to seaborn. 
import seaborn as sns
sns.set_style("whitegrid");
sns.FacetGrid(df,) \
   .map(plt.scatter, "Balance", "Days_since_enroll") \
   .add_legend();
plt.show();

# Notice that the blue points can be easily seperated 

sns.set_style("whitegrid");
sns.FacetGrid(df,) \
   .map(plt.scatter, "Bonus_miles", "Bonus_trans") \
   .add_legend();
plt.show();

# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
# Can be used when number of features are high.

sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()


# Scatterplot on dataframe
sns.scatterplot(data=df)

# mean
df.mean()

# median
df.median()

# Standard deviation
df.std()
''' Standard deviation of the Balance and Bonus_miles is more '''

# Identify the duplicates
 
duplicate=df.duplicated()
# Output of this function is single columns

# if there is duplicate records output- True

# if there is no duplicate records output-False

# Series will be created

duplicate
sum(duplicate)
# Sum is 0 , there is no duplicate data

# IQR
IQR=df.Balance.quantile(0.75)-df.Balance.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR

# Winsorizer 
# This technique is used because no data loss in this techniques.

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Balance']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(df[['Balance']])

sns.boxplot(df[['Balance']])
# Check boxplot of prious one having the outliers

sns.boxplot(df_t['Balance'])
# We can see the outliers are removed


#################################################

# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization
# drop ID#
# We Know that there is scale diff. among the columns, which romove by using 
# Normalization or standasdization

df=df.drop(['ID#'],axis=1)

# Apply Normalization function 

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# whenever there is mixed data apply normalization
# Now apply this normalization function to df for all the rows

df_norm=norm_fun(df.iloc[:,:])

# all data from is up to 1
b=df_norm.describe()
print(b)

# Ideal cluster 
# Defined the number of clusters 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans

a=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    a.append(kmeans.inertia_)

# total within sum of square


print(a)
# As k value increases the a the a value decreases
plt.plot(k,a,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")
'''
How to select value of k from elbow curve
when k changes from 2 to 3 , then decrease
in a is higher than 
when k chages from 3 to 4
when k changes from 3 to 4.
Whwn k value changes from 5 to 6 decreases
in a is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in a is considerably less , hence considered k=3
'''

model=KMeans(n_clusters=3)

model.fit(df_norm)
model.labels_

mb=pd.Series(model.labels_)

df['clust']=mb
df.head()
df=df.iloc[:,[7,0,1,2,3,4,5,6]]
df
df.iloc[:,2:8].groupby(df.clust).mean()

df.to_csv("Airlines.csv",encoding="utf-8")
import os
os.getcwd()

###############################################################