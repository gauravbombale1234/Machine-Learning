# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:21:53 2024

@author: Gaurva Bombale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

forest=pd.read_csv('forestfires.csv')

forest.dtypes

############## EDA
forest.shape

plt.figure(1,figsize=(16,10))
sns.countplot(forest.month)
#Aug and sept has highest values
sns.countplot(forest.day)
#Friday , sunday and saturday has highest value

sns.displot(forest.FFMC)
#data is normal and slight leftskewed
sns.boxplot(forest.FFMC)
#There are several outliers

sns.displot(forest.DMC)
#data is normal and slight right skewed
sns.boxplot(forest.DMC)
#There are several outliers

sns.displot(forest.DC)
#data is normal and slight left skewed
sns.boxplot(forest.DC)
#There are outliers

sns.displot(forest.ISI)
#data is normal 
sns.boxplot(forest.ISI)
#There are outliers

sns.displot(forest.temp)
#data is normal 
sns.boxplot(forest.temp)
#There are outliers

sns.displot(forest.RH)
#data is normal and slight leftskewed
sns.boxplot(forest.RH)
#There are several outliers

sns.displot(forest.wind)
#data is normal and slight right skewed
sns.boxplot(forest.wind)
#There are several outliers

sns.count(forest.rain)
#data is normal 
sns.boxplot(forest.rain)
#There are several outliers

sns.displot(forest.area)
#data is normal 
sns.boxplot(forest.area)
#There are outliers

#Now let  us check the Highest fire in KM?
forest.sort_values(by='area',ascending=False).head(5)

highest_fire_area=forest.sort_values(by='area',ascending=True)

plt.figure(figsize=(8,6))

plt.title('Temperature vs area of fire')

plt.bar(highest_fire_area['temp'],highest_fire_area['area'])

plt.xlabel('Temperature')
plt.ylabel('Area per km-sq')
plt.show()
#once the fire starts , almost 1000+ sq area's temp goes beyond 25 and
#around 750 km area is facing temp 30+
#Now let us check the highest rain in the forest
highest_rain=forest.sort_values(by='rain',ascending=False)[['month','day','rain']].head(5)
highest_rain
#highest rain observed in the month of Aug
#Let us check highest and lowest temp in month and day
highest_temp=forest.sort_values(by='temp',ascending=False)[['month','day','temp']].head(5)

lowest_temp=forest.sort_values(by='temp',ascending=True)[['month','day','temp']].head(5)

print('Highest Temperature ',highest_temp)
#Highest temp observed in Aug
print('Lowest temp ',lowest_temp)
#Lowest temp in the month of Dec

forest.isna().sum()
#There is no missing values in both

#####################################

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size_category)

forest.dtypes

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['month']
                  )
df_t=winsor.fit_transform(forest[['month']])
sns.boxplot(df_t.month)

######################### write  for each column

tc=forest.corr()
tc
fig,ax=plt.subplot()
fig.set_size_inches(200,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#all the varibles are moderately correlated with size_category

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test=train_test_split(forest,test_size=0.3)
train_X=train.iloc[:,:30]
train_y=train.iloc[:,30]
test_X=train.iloc[:,:30]
test_y=train.iloc[:,30]

#Kernal linear
model_linear=SVC(kernel='linear')
model_linear.fit(train_X,train_y)
pred_test_linear=model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)

#RBF
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)

