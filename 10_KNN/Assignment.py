# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:01:45 2024

@author: Gaurav Bombale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
glass=pd.read_csv("glass.csv")
#####################
#(EDA):
glass.dtypes
glass.columns
glass.describe()

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
plt.boxplot(glass.K)
plt.boxplot(glass.Ca)
plt.boxplot(glass.Fe)
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
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
##########
#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)








####################################################### 
########### 2 
"""
A National Zoopark in India is dealing with the problem of segregation
 of the animals based on the different attributes they have. 
 Build a KNN model to automatically classify the animals. 
Explain any inferences you draw in the documentation

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
zoo=pd.read_csv("Zoo.csv")
#####################
#	Exploratory Data Analysis (EDA):
zoo.dtypes
#All the inout features are of float type and output column i.e.type is integer type
zoo.columns
zoo.describe()
plt.hist(zoo.hair)

plt.hist(zoo.feathers)

#############

#Since it is Multivariate, in the type column,only class numbers have been given
#let us change the type of class it belongs
zoo['type']=np.where(zoo['type']=='1','cat-1',zoo['type']) 
zoo['type']=np.where(zoo['type']=='2','cat-2',zoo['type'])
zoo['type']=np.where(zoo['type']=='3','cat-3',zoo['type'])
zoo['type']=np.where(zoo['type']=='4','cat-4',zoo['type'])
zoo['type']=np.where(zoo['type']=='5','cat-5',zoo['type'])
zoo['type']=np.where(zoo['type']=='6','cat-6',zoo['type'])
zoo['type']=np.where(zoo['type']=='7','cat-7',zoo['type'])
zoo.type

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
zoo.shape
#The first column is a animal name let us ignore it,there for 1:16
zoo_norm=norm_func(zoo.iloc[:,1:17])
zoo_norm.describe()

X=np.array(zoo_norm.iloc[:,:])
y=np.array(zoo['type'])
############################
#let us split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
###########################
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Accuracy is 0.9523809523809523
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
##########
#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)
#0.0.9875
pd.crosstab(pred_train,y_train,rownames=['Actual'],colnames=['predicted'])
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
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred

from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Accuracy is 0.9523809523809523
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
##########
#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)

