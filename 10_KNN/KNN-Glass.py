# -*- coding: utf-8 -*-
"""
A glass manufacturing plant uses different earth
 elements to design new glass materials based on
 customer requirements. For that, they would like
 to automate the process of classification as it’s 
 a tedious job to manually classify them. 
 Help the company achieve its objective by correctly 
 classifying the glass type based on the other features using KNN algorithm.
1.	Business Problem
1.1.	What is the business objective?
      1.1.1 Glass production still faces the challeges of finding optimum contents
       for reducing atmospheric emmission.
       1.1.2 Identifying ,reducing and replacing the hazardous substances in the purchased
       material that end up with end product.
1.2.	Are there any constraints?
       1.2.1 Issue of climate change and energy consumtions
             are the major constraints

@author: Gaurav Bombale
"""
#Data Description
#Data Set Characteristics:  Multivariate
#Number of Instances:214
#1. Id number: 1 to 214
#2. RI: refractive index
#3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
#4. Mg: Magnesium
#5. Al: Aluminum
#6. Si: Silicon
#7. K: Potassium
#8. Ca: Calcium
#9. Ba: Barium
#10. Fe: Iron
#11. Type of glass: (class attribute)
#Glass Type1 building_windows_float_processed
#Glass Type 2 building_windows_non_float_processed
#Glass Type 3 vehicle_windows_float_processed
#Glass Type 4 vehicle_windows_non_float_processed (none in this database)
#Glass Type 5 containers
#Glass Type 6 tableware
#Glass Type 7 headlamps
#######################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
glass=pd.read_csv("glass.csv")
#####################
#4.	Exploratory Data Analysis (EDA):
glass.dtypes
#All the inout features are of float type and output column i.e.type is integer type
glass.columns
glass.describe()
#The minimum value of RI is 1.511 and max is 1.53
#The average value of RI is 1.51
#The minimum value of Na is 10.73 and max is 17.38
#The average value of RI is 13.40
plt.hist(glass.RI)
#RI is normally distributed
plt.hist(glass.Na)
#Na is normally distributed
plt.hist(glass.Mg)
#Mg is left skewed normal distributed
plt.hist(glass.Al)
#Al data is normally distributed
#############
#let us check the outliers in the dataset
plt.boxplot(glass.Si)
#There are several outliers in Si
plt.boxplot(glass.K)
#There are several ouliers in K
plt.boxplot(glass.Ca)
#There are several outliers in the Ca data
plt.boxplot(glass.Fe)
#There are several outliers in the Fe data
#######################################
#3.	Data Pre-processing
from feature_engine.outliers import Winsorizer
import seaborn as sns
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RI'])
glass_t=winsor.fit_transform(glass[['RI']])
sns.boxplot(glass_t.RI)
glass.var()
#####################
sns.boxplot(glass.Na)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Na'])
glass_t=winsor.fit_transform(glass[['Na']])
sns.boxplot(glass_t.Na)
###################
sns.boxplot(glass.Mg)
#There are no ouliers in Mg
########################3
sns.boxplot(glass.Al)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Al'])
glass_t=winsor.fit_transform(glass[['Al']])
sns.boxplot(glass_t.Al)
###################
sns.boxplot(glass.Si)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Si'])
glass_t=winsor.fit_transform(glass[['Si']])
sns.boxplot(glass_t.Si)
######################
sns.boxplot(glass.K)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['K'])
glass_t=winsor.fit_transform(glass[['K']])
sns.boxplot(glass_t.K)
###############################
sns.boxplot(glass.Ca)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Ca'])
glass_t=winsor.fit_transform(glass[['Ca']])
sns.boxplot(glass_t.Ca)
#######################
sns.boxplot(glass.Fe)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Fe'])
glass_t=winsor.fit_transform(glass[['Fe']])
sns.boxplot(glass_t.Fe)
##########################
#Since it is Multivariate, in the type column,only class numbers have been given
#let us change the type of class it belongs
glass['Type']=np.where(glass['Type']=='1','build_win_fl',glass['Type']) 
glass['Type']=np.where(glass['Type']=='2','build_win_nfl',glass['Type'])
glass['Type']=np.where(glass['Type']=='3','veh_win_fl',glass['Type'])
glass['Type']=np.where(glass['Type']=='4','veh_win_nfl',glass['Type'])
glass['Type']=np.where(glass['Type']=='5','containers',glass['Type'])
glass['Type']=np.where(glass['Type']=='6','tableware',glass['Type'])
glass['Type']=np.where(glass['Type']=='7','headlamps',glass['Type'])
glass.Type
###############################################
#All the columns having data in different scales ,hence need normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
glass_norm=norm_func(glass.iloc[:,:9])
glass_norm.describe()
#################################################
# Training the model
#Before that,let us assign input and output columns
X=np.array(glass_norm.iloc[:,:])
y=np.array(glass['Type'])
############################
#let us split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
###########################
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Accuracy is 0.46511627906976744
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
##########
#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)
#0.6608187134502924
pd.crosstab(pred_train,y_train,rownames=['Actual'],colnames=['predicted'])
##############################################
#Tunning of the model
#For selection of optimum value of k
acc=[]
#Let us run KNN on values 3,50 in step of 2 so that next value will be odd
for i in range(3,50,2):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X_train,y_train)
    train_acc=np.mean(knn1.predict(X_train)==y_train)
    test_acc=np.mean(knn1.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
#To plot the graph of accuracy of training and testing
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")    
plt.plot(np.arange(3,50,2),[i[1]for i in acc],"bo-")
#K=3 has got better accuracy
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Accuracy is 0.46511627906976744
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
##########
#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)
#still the model is over fit
