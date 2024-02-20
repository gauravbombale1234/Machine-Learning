# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:27:37 2023

@author: Gaurav Bombale
"""
####################################################################
############   one hot encoder  ####################################
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
# we use ethnic diversity datase
df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")
df.columns
# we have Salaries and age as numerical columns , let us make them
# at position 0 and 1 so to make further dataprocessing easy
df=df[['Salaries','age','Employee_Name', 'Position', 'State', 'Sex',
       'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department', 'Race']]
#check the dataframe in the variable explorer
# we want only nominal data and ordinal data for processing
# hence skipped 0th and 1st column and applied to one hot encoder
enc_df=pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())
#label encoder


####################################################################
############   Label encoder  ####################################

import pandas as pd
from sklearn.preprocessing import LabelEncoder
#creating instance of label encoder
labelencoder=LabelEncoder()
# split your data into input and output variables
X=df.iloc[:,0:9]    #first 8 columns for X and 9th for y
y=df['Race']
df.columns
# we have nominal data Sex,MaritalDesc,CitizenDesc,
# we want to convert to label encoder
X['Sex']=labelencoder.fit_transform(X['Sex'])
X['MaritalDesc']=labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc']=labelencoder.fit_transform(X['CitizenDesc'])
# label encoder y
y=labelencoder.fit_transform(y)
# this is going to create an array , hence convert
# it back to DataFrame
y=pd.DataFrame(y)
df_new=pd.concat([X,y],axis=1)
# if you will see variable explorer , y do not have column name
# hence rename the column
df_new=df_new.rename(columns={0:'Race'})


############################################################
#####  Standardization and  Normalization  ########
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("C:/Datasets/mtcars.csv")
d.describe()
a=d.describe()
# initialize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()
# here if you will check res,in variable environment then
# 

############################ for Seeds_data.csv dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("C:/Datasets/Seeds_data.csv")
d.describe()
a=d.describe()
# initialize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()

#########################################
############ Normalization 
ethnic=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")
# now read columns
ethnic.columns
# there are some columns which not usefull , we need to drop
ethnic.drop(['Employee_Name','EmpID','Zip'],axis=1,inplace=True)
# now read the minimum value and maximum values of Salaries and age
a1=ethnic.describe()
# check a1 data frame in variable explorer ,
# you find minimum salary is 0  and max is 108304
# same way check for age, there is huge difference
# in min and max. value . Hence we are going for normalization.
# first we will have t o convert non-numeric data to label encoding
ethnic=pd.get_dummies(ethnic,drop_first=True)
# normalization function  written where ehnic argument is passed
def norm_func(i):
    x=(i-i.min())//(i.max()-i.min())
    return x
df_norm=norm_func(ethnic)
b=df_norm.describe()
# if you will observe the b frame,
# it has dimention 8,81
# earlier in a they were 8,11, it is because all non-numeric
# data has been converted to numeric using label encoding

