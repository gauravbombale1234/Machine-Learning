# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:04:50 2024

@author: Gaurav Bombale
"""

""" Problem Statement:
2.	Most cancers form a lump called a tumour. But not all lumps are cancerous. 
Doctors extract a sample from the lump and examine it to find out if it’s 
cancer or not. Lumps that are not cancerous are called benign (be-NINE). 
Lumps that are cancerous are called malignant (muh-LIG-nunt). Obtaining 
incorrect results (false positives and false negatives) especially in a 
medical condition such as cancer is dangerous. So, perform Bagging, Boosting, 
Stacking, and Voting algorithms to increase model performance and provide your 
insights in the documentation.


1.	Business Problem
1.1.	What is the business objective?
  1.1.1 The fundamental goals of cancer prediction and prognosis are distinct from the goals of cancer detection
and diagnosis. In cancer prediction/prognosis one is concerned with three predictive foci: 1) the prediction of
cancer susceptibility (i.e. risk assessment); 2) the prediction of cancer recurrence and 3) the prediction of
cancer survivability.
  1.1.2 Every year, Pathologists diagnose 14 million new patients
  with cancer around the world. That’s millions of people who’ll 
  face years of uncertainty.There are chances of misclassification
  
1.1.	Are there any constraints?
    The main components in a standard ML based system are 
    preprocessing, feature recognition, extraction and selection, 
    categorization, and performance assessment.

"""
#diagnosis              int32
#radius_mean          float64
#texture_mean         float64
#perimeter_mean       float64
#area_mean            float64
#smoothness_mean      float64
#compactness_mean     float64
#concavity_mean       float64
#points_mean          float64
#symmetry_mean        float64
#dimension_mean       float64
#radius_se            float64
#texture_se           float64
#perimeter_se         float64
#area_se              float64
#smoothness_se        float64
#compactness_se       float64
#concavity_se         float64
#points_se            float64
#symmetry_se          float64
#dimension_se         float64
#radius_worst         float64
#texture_worst        float64
#perimeter_worst      float64
#area_worst           float64
#smoothness_worst     float64
#compactness_worst    float64
#concavity_worst      float64
#points_worst         float64
#symmetry_worst       float64
#dimension_worst      float64
####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("C:/11_Ensemble Learning/Tumor_Ensemble.csv")
df.dtypes
df.describe()
#average radius_mean of tumar is 14.12
#min is 0 and max is 6.98
#Avearge symmetry_worst is 0.29
#min is 21 and max is 0.15
###########################
##The column 0 is not useful
df.drop(["id"],axis=1,inplace=True)
###EDA
df.dtypes
plt.hist(df.radius_mean)
#Data is normally distributed 
plt.hist(df.texture_mean)
#Data is normally distributed
plt.hist(df.perimeter_mean)
#Data is normally distributed but right skewed
plt.hist(df.area_mean)
# data is apparently normal distributed but right skewed
plt.hist(df.smoothness_mean)
#Data is normally distributed
plt.hist(df.compactness_mean)
#Data is normally distributed but right skewed
#let us check the outliers
plt.boxplot(df.radius_mean)
#There are outliers
plt.boxplot(df.texture_mean)
#There are outliers
plt.boxplot(df.perimeter_mean)
#There are outliers
plt.boxplot(df.area_mean)
#There are outliers
plt.boxplot(df.smoothness_mean)
#There are outliers
plt.boxplot(df.compactness_mean)
#There are outliers
plt.boxplot(df.concavity_mean )
#There are outliers
plt.boxplot(df.points_mean)
#There are outliers
df.dtypes

plt.boxplot(df.symmetry_mean)
#There are outliers
plt.boxplot(df.dimension_mean)
#There are outliers
plt.boxplot(df.radius_se)
#There are outliers
plt.boxplot(df.texture_se )
#There are outliers
plt.boxplot(df.perimeter_se )
#There are outliers
plt.boxplot(df.area_se)
#There are outliers
plt.boxplot(df.smoothness_se  )
#There are outliers
plt.boxplot(df.compactness_se)
#There are outliers

plt.boxplot(df.concavity_se)
#There are outliers
plt.boxplot(df.points_se )
#There are outliers
plt.boxplot(df.symmetry_se )
#There are outliers
plt.boxplot(df.dimension_se )
#There are outliers
plt.boxplot(df.radius_worst )
#There are outliers
plt.boxplot(df.texture_worst)
#There are outliers
plt.boxplot(df.perimeter_worst  )
#There are outliers
plt.boxplot(df.area_worst )
#There are outliers

plt.boxplot(df.smoothness_worst)
#There are outliers
plt.boxplot(df.compactness_worst )
#There are outliers
plt.boxplot(df.concavity_worst )
#There are outliers
plt.boxplot(df.points_worst )
#There are no outliers
plt.boxplot(df.symmetry_worst )
#There are outliers
plt.boxplot(df.dimension_worst)
#There are outliers

df.isnull().sum()
#There are no null values
##########################################
###Data preprocessing
df.dtypes
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["radius_mean"])
df_t=winsor.fit_transform(df[["radius_mean"]])
sns.boxplot(df_t.radius_mean)
########
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["texture_mean"])
df_t=winsor.fit_transform(df[["texture_mean"]])
sns.boxplot(df_t.texture_mean)
##############
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["perimeter_mean"])
df_t=winsor.fit_transform(df[["perimeter_mean"]])
sns.boxplot(df_t.perimeter_mean)
#########
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["area_mean"])
df_t=winsor.fit_transform(df[["area_mean"]])
sns.boxplot(df_t.area_mean)
#####
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["smoothness_mean"])
df_t=winsor.fit_transform(df[["smoothness_mean"]])
sns.boxplot(df_t.smoothness_mean)
###
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["compactness_mean"])
df_t=winsor.fit_transform(df[["compactness_mean"]])
sns.boxplot(df_t.compactness_mean)
#####
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["concavity_mean"])
df_t=winsor.fit_transform(df[["concavity_mean"]])
sns.boxplot(df_t.concavity_mean)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["points_mean"])
df_t=winsor.fit_transform(df[["points_mean"]])
sns.boxplot(df_t.points_mean)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["symmetry_mean"])
df_t=winsor.fit_transform(df[["symmetry_mean"]])
sns.boxplot(df_t.symmetry_mean )
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["dimension_mean"])
df_t=winsor.fit_transform(df[["dimension_mean"]])
sns.boxplot(df_t.dimension_mean)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["radius_se"])
df_t=winsor.fit_transform(df[["radius_se"]])
sns.boxplot(df_t.radius_se )
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["texture_se"])
df_t=winsor.fit_transform(df[["texture_se"]])
sns.boxplot(df_t.texture_se )
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["perimeter_se"])
df_t=winsor.fit_transform(df[["perimeter_se"]])
sns.boxplot(df_t.perimeter_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["area_se"])
df_t=winsor.fit_transform(df[["area_se"]])
sns.boxplot(df_t.area_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["smoothness_se"])
df_t=winsor.fit_transform(df[["smoothness_se"]])
sns.boxplot(df_t.smoothness_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["compactness_se"])
df_t=winsor.fit_transform(df[["compactness_se"]])
sns.boxplot(df_t.compactness_se )
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["concavity_se"])
df_t=winsor.fit_transform(df[["concavity_se"]])
sns.boxplot(df_t.concavity_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["points_se"])
df_t=winsor.fit_transform(df[["points_se"]])
sns.boxplot(df_t.points_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["symmetry_se"])
df_t=winsor.fit_transform(df[["symmetry_se"]])
sns.boxplot(df_t.symmetry_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["dimension_se"])
df_t=winsor.fit_transform(df[["dimension_se"]])
sns.boxplot(df_t.dimension_se)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["radius_worst"])
df_t=winsor.fit_transform(df[["radius_worst"]])
sns.boxplot(df_t.radius_worst)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["texture_worst"])
df_t=winsor.fit_transform(df[["texture_worst"]])
sns.boxplot(df_t.texture_worst)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["perimeter_worst"])
df_t=winsor.fit_transform(df[["perimeter_worst"]])
sns.boxplot(df_t.perimeter_worst)
##################
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["area_worst"])
df_t=winsor.fit_transform(df[["area_worst"]])
sns.boxplot(df_t.area_worst)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["smoothness_worst"])
df_t=winsor.fit_transform(df[["smoothness_worst"]])
sns.boxplot(df_t.smoothness_worst)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["compactness_worst"])
df_t=winsor.fit_transform(df[["compactness_worst"]])
sns.boxplot(df_t.compactness_worst)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["concavity_worst"])
df_t=winsor.fit_transform(df[["concavity_worst"]])
sns.boxplot(df_t.concavity_worst)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["points_worst"])
df_t=winsor.fit_transform(df[["points_worst"]])
sns.boxplot(df_t.points_worst)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["symmetry_worst"])
df_t=winsor.fit_transform(df[["symmetry_worst"]])
sns.boxplot(df_t.symmetry_worst)
##################
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["dimension_worst"])
df_t=winsor.fit_transform(df[["dimension_worst"]])
sns.boxplot(df_t.dimension_worst )
##################

#There are several columns having different scale,hence let us apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df.iloc[:,1:31])
######################
#let us check how many unique values are there in outcome
df["diagnosis"].unique()
df["diagnosis"].value_counts()
################
predictors=df.iloc[:,df.columns!="diagnosis"]
target=df["diagnosis"]
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
df["diagnosis"]=lb.fit_transform(df["diagnosis"])
predictors=df.iloc[:,df.columns!="diagnosis"]
target=df["diagnosis"]
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
y=df['diagnosis']
train_x,train_y=x.iloc[:400],y.iloc[:400]
test_x,test_y=x.iloc[400:],y.iloc[400:]
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
##voting
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

