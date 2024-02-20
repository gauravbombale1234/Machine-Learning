# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:40:02 2023

@author: Gaurav Bombale
"""
import pandas as pd

df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")

df.dtypes

# convert salaries to int
df.Salaries=df.Salaries.astype(int)
df.dtypes

#convert data type of age to float 
df.age=df.age.astype(float)
df.dtypes

### identify the duplicates
df_new=pd.read_csv("C:/3-CRISP-ML(Q)/education.csv")
duplicate=df_new.duplicated()
'''
output of this function is single column 
if there is duplicate records output - True
if there is no duplicate records output - False
Series will be created
'''
duplicate
sum(duplicate)

####################
df_new1=pd.read_csv("C:/3-CRISP-ML(Q)/mtcars_dup.csv")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
'''
there are 3 duplicate records
row 17 is duplicate of row 2 like wise you can 3 duplicate records

'''
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)

###################################
##### OUTLIERS TREATMENT
import pandas as pd
import seaborn as sns

df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")

sns.boxplot(df.Salaries)
#there are outliers
#let us check outlier in age column
sns.boxplot(df.age)
#there are no outliers
# let us calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observed IQR in variable explorer
# no because IQR is in capital letters
# treated as constant
IQR
#but if we will try as I , Iqr or iqr then it is showing
# I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR

upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#now if you will check the lower limit of 
# salary , it is -19446.9675
# there is negetive salary
# so make it as 0
# how to make it 
# go to variable explorer and make it 0


##############################
### Trimming
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df_trimmed=df.loc[~outliers_df]
df.shape
df_trimmed.shape

###############################
#### Replacement Technique
# drawback of trimming technique is we are loosing the data
df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")
df.describe()

# record no. 23 has got outliers
# map all the outlier  values to upper limit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
#if the values are greater than upper limit
# map it to upper limit and less than lower limit
# map it to lower limit , if it within the range
# then keep as it is 
sns.boxplot(df_replaced[0])


"""
Created on Fri Oct  6 08:37:53 2023

@author: Gaurav Bombale
"""
#### Winsorizer
df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")
import seaborn as sns

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries'])

#copy winsorizer and paste in help tab of top right window, study the method
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])


"""
Created on Mon Oct  9 08:32:57 2023

@author: Gaurav Bombale
"""
'''
Zero variance and  near zero variance
if there is no variance in the feature , then ML model
will not get any intelligence, so it is better to ignore those featuress
'''
import pandas as pd 

df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")
df.dtypes
df.var()

# or 
df.var()==0

df.var(axis=0)==0


##############################
import pandas as pd 
import numpy as np

df=pd.read_csv("C:/Datasets/modified ethnic.csv")
# check for null values
df.isna().sum()

######################################
## create an inputer that creates NaN values

from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
# check the dataframe
df['Salaries']=pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))
df['Salaries'].isna().sum()


"""
Created on Mon Oct  9 15:33:57 2023

@author: Gaurav Bombale
"""
import pandas as pd 
import numpy as np
data=pd.read_csv("C:/Datasets/modified ethnic.csv")
data.head(10)
data.info()


data.describe()
data['Salaries_new']=pd.cut(data['Salaries'],bins=[min(data.Salaries),data.Salaries.mean(),max(data.Salaries)],labels=["low","High"])
data.Salaries_new.value_counts()

data['Salaries_new']=pd.cut(data['Salaries'],bins=[min(data.Salaries),data.Salaries.quantile(0.25),data.Salaries.mean(),data.Salaries.quantile(0.75),max(data.Salaries)],labels=["group1","group2","group3","group4"])
data.Salaries_new.value_counts()

###############################################################
###############################################################
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Datasets/animal_category.csv")
df.shape
df.drop(['Index'],axis=1,inplace=True)
#check df again
df_new=pd.get_dummies(df)
df_new.shape
#here we are getting 30 rows and 14 columns
#we are getting two columns for homely and gender , one column for each is
#delete second column of gender and second column of homely
df_new.drop(["Gender_Male","Homly_Yes"],axis=1,inplace=True)
df_new.shape
# now we are getting 30,12
df_new.rename(columns={'Gende_Female':'Gender','Homly_No':'Homly'})


####
df=pd.read_csv("C:/3-CRISP-ML(Q)/ethnic diversity.csv")
df.shape

df.head()

#df.drop(['Index'],axis=1,inplace=True)

df_new=pd.get_dummies(df)

df_new.shape

df_new.drop(['EmpID','Zip','Salaries','age'],axis=1,inplace=True)

df_new.shape
