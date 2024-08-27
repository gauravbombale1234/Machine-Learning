# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:04:50 2024

@author: Gaurav Bombale
"""

""" Problem Statement:
1.	Given is the diabetes dataset.
 Build an ensemble model to correctly classify the outcome variable 
 and improve your model prediction by using GridSearchCV. 
 You must apply Bagging, Boosting, Stacking, and Voting on the dataset.  


1.	Business Problem
1.1.	What is the business objective?
  1.1.1 With the development of living standards, diabetes is increasingly
 common in people’s daily life. Therefore, how to quickly and accurately
 diagnose and analyze diabetes is a topic worthy studying. 
 
 1.1.2 In medicine, the diagnosis of diabetes is according to fasting blood glucose,
 glucose tolerance, and random blood glucose levels. 
 The earlier diagnosis is obtained, the much easier we can control it. Machine 
 learning can help people make a preliminary judgment about diabetes mellitus 
 according to their daily physical examination data, and it can serve as a 
 reference for doctors
  
1.1.	Are there any constraints?
    For machine learning method, how to select the valid features 
and the correct classifier are the most important constraints.
Several constraints were placed on the selection of these 
instances from a larger database. In particular, all patients
here are females at least 21 years old of Pima Indian heritage.
"""
#Data description
#Pregnancies:Number of times pregnant int64	
#Glucose	:Plasma glucose concentration int64
#BloodPressure	:Diastolic blood pressure int64
#SkinThickness	:Triceps skin fold thickness  int64
#Insulin	:2-Hour serum insulin int64
#BMI	    :Body mass index  float
#DiabetesPedigreeFunction	
#Age	
#Outcome
####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("C:/11_Ensemble Learning/Diabeted_Ensemble.csv")
df.dtypes
df.describe()
#average number of times pregnant is 3.8
#min is 0 and max is 17.00
#Avearge age is 33.24 
#min is 21 and max is 81
###########################
##The column names renaming
new_names=["Pregnancies","Glucose","BP","Skin_thickness","Insulin","BMI","D_pedigree","Age","Outcome"]
df=pd.read_csv("C:/360DG/Datasets/Diabeted_Ensemble.csv",names=new_names, header=0,usecols=[0,1,2,3,4,5,6,7,8])

###EDA
df.dtypes
plt.hist(df.Pregnancies)
#Data is normally distributed but right skewed
plt.hist(df.Glucose)
#Data is normally distributed
plt.hist(df.BP)
#Data is normally distributed
plt.hist(df.Skin_thickness)
# data is apparently normal distributed
plt.hist(df.Insulin)
#Data is normally distributed
plt.hist(df.BMI)
#Data is normally distributed
#let us check the outliers
plt.boxplot(df.Pregnancies)
#There are outliers
plt.boxplot(df.Glucose)
#There are outliers
plt.boxplot(df.BP)
#There are outliers
plt.boxplot(df.Skin_thickness)
#There are outliers
plt.boxplot(df.Insulin)
#There are outliers
plt.boxplot(df.BMI)
#There are outliers
plt.boxplot(df.D_pedigree)
#There are outliers
plt.boxplot(df.Age)
#There are outliers
df.isnull().sum()
#There are no null values
##########################################
###Data preprocessing
df.dtypes
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Pregnancies"])
df_t=winsor.fit_transform(df[["Pregnancies"]])
sns.boxplot(df_t.Pregnancies)
########
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Glucose"])
df_t=winsor.fit_transform(df[["Glucose"]])
sns.boxplot(df_t.Glucose)
##############
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["BP"])
df_t=winsor.fit_transform(df[["BP"]])
sns.boxplot(df_t.BP)
#########
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Skin_thickness"])
df_t=winsor.fit_transform(df[["Skin_thickness"]])
sns.boxplot(df_t.Skin_thickness)
#####
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Insulin"])
df_t=winsor.fit_transform(df[["Insulin"]])
sns.boxplot(df_t.Insulin)
###
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["BMI"])
df_t=winsor.fit_transform(df[["BMI"]])
sns.boxplot(df_t.BMI)
#####
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["D_pedigree"])
df_t=winsor.fit_transform(df[["D_pedigree"]])
sns.boxplot(df_t.D_pedigree)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Age"])
df_t=winsor.fit_transform(df[["Age"]])
sns.boxplot(df_t.Age)
##################
#There are several columns having different scale,hence let us apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df.iloc[:,:8])
######################
#let us check how many unique values are there in outcome
df["Outcome"].unique()
df["Outcome"].value_counts()
################
predictors=df.iloc[:,df.columns!="Outcome"]
target=df["Outcome"]
#################
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
ada_boost=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)
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
df["Outcome"]=lb.fit_transform(df["Outcome"])
predictors=df.iloc[:,df.columns!="Outcome"]
target=df["Outcome"]
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
y=df['Outcome']
train_x,train_y=x.iloc[:600],y.iloc[:600]
test_x,test_y=x.iloc[600:],y.iloc[600:]
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

#what way does the business (client) benefit from the solution provided?
#This  will portray how data related to diabetes can be leveraged
# to predict if a person has diabetes or not. More specifically, this will focus on how machine learning 
# can be utilized to predict diseases such as diabetes.
#The use of decision tree-based data mining to establish prediction of adiabetics is advantageous
 #because it can (1) allow for a wider coverage of features matrix with a fewer number of steps, 
 #(2) generate regular predictions patterns and rules, and (3) provide doctors with reference points to facilitate the treatment. 
 #The newly developed sales system can provide garment manufacturers with insights, design development, pattern grading, and market analysis.
 #Moreover, when production plans can be made more realistic, inventory costs due to mismatches can be minimized
