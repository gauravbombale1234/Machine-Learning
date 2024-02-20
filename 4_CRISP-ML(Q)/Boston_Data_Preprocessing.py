# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:56:02 2023

@author: Gaurav Bombale
"""
# Boston dataset Datapreprocessing
'''
Business Objective:
    
    Minimize: the housing rate 
    Maximize: the accuracy of the house price prediction

Continuous Data 
'''
import seaborn as sns
import pandas as pd

df=pd.read_csv("C:/3-CRISP-ML(Q)/Boston.csv")

df.dtypes
df.describe()

sns.boxplot(df.crim)
sns.boxplot(df.zn)
sns.boxplot(df.indus)   # no outliers
sns.boxplot(df.chas)



### identify the duplicates
df_new=pd.read_csv("C:/3-CRISP-ML(Q)/Boston.csv")
duplicate=df_new.duplicated()
duplicate
sum(duplicate)
