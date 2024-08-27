# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:04:51 2024

@author: Gaurav Bombale
"""

""" Problem Statement:
3.	A sample of global companies and their ratingsare given for the 
cocoa bean production along with the location of the beans being used.
Identify the important features in the analysis and accurately classify 
the companies based on their ratings and draw insights from the data. 
Build ensemble models such as Bagging, Boosting, Stacking, and Voting on 
the dataset given.


1.	Business Problem
1.1.	What is the business objective?
  1.1.1 This analysis will bring some insight into consumer patterns in 
  the chocolate industry. It will be possible to perhaps see a pattern
  in the data that could lead us to observe why a specific amount of cocoa 
  is used in chocolate bars and how it will affect consumer rating.
  Finally, it will show us how geographic factors have an impact on the 
  consumption of chocolate,and where the highest rated chocolate and cocoa 
  beans come from.
  1.1.2 
  
1.1.	Are there any constraints?
    
"""
#Company              object
#Name                 object
#REF  float64 A value linked to when the review was 
#entered in the database. Higher = more recent
#Review              float64
#Cocoa_Percent       float64
#Company_Location     object
#Rating              float64
#Bean_Type            object
#Origin               object
####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel("C:/11_Ensemble Learning/Coca_Rating_Ensemble.xlsx")
df.dtypes
df.describe()
#average REF is 1035.90
#min is 5 and max is 1952
#Avearge Review is 2012.32
#min is 2006 and max is 2017
#The average percent of cocoa in a chocolate bar 
#is around 70%, and the average rating that 
#consumers are giving to chocolate bars is 3.1
###########################

###EDA
df.dtypes
plt.hist(df.REF)
#Data is normally distributed 
plt.hist(df.Review)
#Data is normally distributed but left skewed
plt.hist(df.Cocoa_Percent)
#Data is normally distributed but right skewed
plt.hist(df.Rating)
# data is apparently normal distributed but right skewed

#let us check the outliers
plt.boxplot(df.REF)
#There are  no outliers
plt.boxplot(df.Review)
#There are  no outliers
plt.boxplot(df.Cocoa_Percent)
#There are outliers
plt.boxplot(df.Rating)
#There are outliers


df.isnull().sum()
#There are  null values in Bean_type and Origin
##########################################
###Data preprocessing
df.dtypes
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Cocoa_Percent"])
df_t=winsor.fit_transform(df[["Cocoa_Percent"]])
sns.boxplot(df_t.Cocoa_Percent)
########
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Rating"])
df_t=winsor.fit_transform(df[["Rating"]])
sns.boxplot(df_t.Rating)
##################
#First and foremost, it is important to know just how many chocolate bars were rated.
sns.countplot(x='Rating',data=df)
plt.xlabel("Rating")
plt.ylabel("Number of users")
#most of the users have given rating between 3 to 3.5
sns.distplot(a=df["Cocoa_Percent"],hist=True,kde=False,rug=False,color="darkgreen")
#most chocolate bars are made with around 70 to 75% cocoa in the bar. With 
#close to 700 chocolate bars being made with 70% cocoa.
#Let us see top 20 countries producing highest rated chocolates

origin_max=df.groupby(["Company"])["Rating"].max()
top=origin_max.sort_values(ascending=False)
top_20=top.iloc[:20]
top_20
origin_max.describe()

bins = [1,2,3,4,5]
group_name=["poor_rated", "ave_rated","good_rated","top_rated"]
df['Company_rating']=pd.cut(df["Rating"],bins,labels=group_name)



from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
#df["Company"]=lb.fit_transform(df["Company"])
df["Name"]=lb.fit_transform(df["Name"])
df["Company_Location"]=lb.fit_transform(df["Company_Location"])
df["Bean_Type"]=lb.fit_transform(df["Bean_Type"])
df["Origin"]=lb.fit_transform(df["Origin"])


from sklearn.impute import SimpleImputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
df["Bean_Type"]=pd.DataFrame(mode_imputer.fit_transform(df[["Bean_Type"]]))
df["Bean_Type"].isna().sum()

df["Origin"]=pd.DataFrame(mode_imputer.fit_transform(df[["Origin"]]))
df["Origin"].isna().sum()

#There are several columns having different scale,hence let us apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df.iloc[:,1:9])
######################
#let us check how many unique values are there in outcome
df["Company_rating"].unique()
df["Company_rating"].value_counts()
df["Company_rating"].isna().sum()
df["Company_rating"]=pd.DataFrame(mode_imputer.fit_transform(df[["Company_rating"]]))
df["Company_rating"].isna().sum()

################
predictors=df_norm
target=df["Company_rating"]
#################dia
#splitting dataset into train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)
##################
#let us apply to bagging
from sklearn import tree
clftree=tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bag_clf=BaggingClassifier(base_estimator=clftree,n_estimators=500,bootstrap=True,n_jobs=1,random_state=42)
bag_clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix
pred1=bag_clf.predict(x_test)
accuracy_score(y_test,pred1)
confusion_matrix(y_test,pred1)
###################
#Evalution on training data
pred2=bag_clf.predict(x_train)
accuracy_score(y_train,pred2)
#############################################
###Ada Boosting
from sklearn.ensemble import AdaBoostClassifier
ada_boost=AdaBoostClassifier()
ada_boost.fit(x_train,y_train)
pred3=ada_boost.predict(x_test)
accuracy_score(y_test,pred3)
#############
#Evaluation on training data
pred4=ada_boost.predict(x_train)
accuracy_score(y_train,pred4)
#################################################
########Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
grand_boost=GradientBoostingClassifier()
grand_boost.fit(x_train,y_train)
######################
##Evaluation on test data
pred5=grand_boost.predict(x_test)
accuracy_score(y_test,pred5)
##################
#Evalution on train data
pred6=grand_boost.predict(x_train)
accuracy_score(y_train,pred6)
#################################################
####XGBoost
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["Company_rating"]=lb.fit_transform(df["Company_rating"])
#predictors=df.iloc[:,df.columns!="diagnosis"]
target=df["Company_rating"]
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)
import xgboost as xgb
xgb_boost=xgb.XGBClassifier(max_depth=5,n_estimators=5000,training_rate=0.3,n_jobs=-1)
xgb_boost.fit(x_train,y_train)
###################
pred7=xgb_boost.predict(x_test)
accuracy_score(y_test,pred7)
###########
pred8=xgb_boost.predict(x_train)
accuracy_score(y_train,pred8)
######################################
######stacking
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
x=df_norm
y=df['Company_rating']
train_x,train_y=x.iloc[:1500],y.iloc[:1500]
test_x,test_y=x.iloc[1500:],y.iloc[1500:]
base_learners=[]
#let us create base learner1
knn=KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)
###Let us create base learner2
dtr=DecisionTreeClassifier(max_depth=4,random_state=1234)
base_learners.append(dtr)
##################### base learner3
mlpc=MLPClassifier(hidden_layer_sizes=(100,),solver='lbfgs',random_state=12345)
base_learners.append(mlpc)
##############meta learner
meta_learner=LogisticRegression(solver='lbfgs')
##############################
#Now to create meta data
meta_data=np.zeros((len(base_learners),len(x_train)))
meta_targets=np.zeros(len(x_train))
#create cross validation
KF=KFold(n_splits=5)
meta_index=0
for train_indices,test_indices in KF.split(train_x):
    for i in range(len(base_learners)):
        learner=base_learners[i]
        learner.fit(train_x[train_indices],train_y[train_indices])
        predictions=learner.predict_proba(train_x[test_indices])[:,0]
        meta_data[i][meta_index:meta_index+len(test_indices)]=predictions
    meta_targets[meta_index:meta_index+len(test_indices)]=train_y[test_indices] 
    meta_index+=len(test_indices)
    #This meta data is used for training
# in order to evaluate the meta-learner test data is derived
test_meta_data=np.zeros((len(base_learners),len(test_x)))
base_acc=[]
###
#base accuracy
for i in range (len(base_learners)):
    learner=base_learners[i]
    learner.fit(train_x,train_y)
    predictions=learner.predict_proba(test_x)[:,0]
    test_meta_data[i]=predictions
    acc=metrics.accuracy_score(test_y,learner.predict(test_x))
    base_acc.append(acc)
test_meta_data = test_meta_data.transpose()
###################################
#Now now train the meta learner on train set and evaluate on meta_target
meta_learner.fit(meta_data,meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)
acc=metrics.accuracy_score(test_y,ensemble_predictions)
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')



########################################
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model,svm,neighbors,naive_bayes
learner_1=neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2=linear_model.Perceptron(tol=1e-2,random_state=0)
learner_3 = svm.SVC(gamma=0.01)
#Now let us instantiate voting classifier
voting=VotingClassifier([('KNN',learner_1),
                          ('Prc',learner_2),
                          ('SVM',learner_3)

                        ])
#Now let us apply it to the train data set
voting.fit(x_train,y_train)
#predict the most voted class
hard_predictions=voting.predict(x_test)
print("Hard Voting",accuracy_score(y_test,hard_predictions))
##############################
#Soft voting 
learner_4=neighbors.KNeighborsClassifier(n_neighbors=5)
learner_5 = naive_bayes.GaussianNB()
learner_6=svm.SVC(gamma=0.01,probability=True)
voting=VotingClassifier([('KNN',learner_4),
                          ('NB',learner_5),
                          ('SVM',learner_6)],
                           voting='soft'

                         )
# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))
##########################################################3
#6.	Write about the benefits/impact of the solution - in 

#One of the best ways to fight cancer is early detection,
#when it is still confined and can be fully excised surgically 
# or treated pharmacologically. Cancer screening programs, that is, 
# the practice of testing for the presence of cancer in people who 
# have no symptoms, has been medicine’s tool of choice for the 
# earliest detection.
