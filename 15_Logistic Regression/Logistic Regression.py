# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:32:11 2024

@author: Gaurav Bombale
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report

clainmants=pd.read_csv('C:/14_Logistic Regression/claimants.csv')
#There are CLMAGE and LOSS are having continuous data rest are categorical
#verify the dataset , where CASENUM is not really usefull so droping the CASENUM
c1=clainmants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()
#let us check whether there are null values
c1.isna().sum()
#There are several null values
#if we used dropna() function we will loose 290 data points
#hence we will go for imputation
c1.dtypes

mean_value=c1.CLMAGE.mean()
mean_value
#now let us impute the same
c1.CLMAGE=c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
#hence all null values of CLMAGE has been filled by mean value

#for columns where there are discrete values , we will apply mode
mode_CLMSEX=c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX=c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()

#CLMINSUR is also categorical data hence mode imputation is applied
mode_CLMINSUR=c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR=c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()

#SEATBELT is categorical data hence go for mode imputation
mode_SEATBELT=c1.SEATBELT.mode()
mode_SEATBELT
c1.SEATBELT=c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()

#Now the person we met an accident will hire the Atternev or not

#Let us build the model 
logit_model=sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=c1).fit()
logit_model.summary()
#In Logistic Regression we do not have R Squared values, only check p value
#SEATBELT is statistically insignificant ignore and proceed
logit_model.summary2()
#here we are going to check AIC value , it stands for Akaike Information Criterion
#is mathematical method for evalution how well a model fits the data
#A lower the score more the  better model , AIC scores are only usefull in 
#with other AIC scores for the same dataset

#NOw let us go for prediction
pred=logit_model.predict(c1.iloc[:,1:])
#here we are applying all rows columns from 1 , as columns 0 is ATTORNEY
#target value 

#let us check the performance of model
fpr,tpr,thresholds=roc_curve(c1.ATTORNEY,pred)
#we are applying actual values and predicted values so as to  get
#false positive rate , true positive rate and threshold
#The Optimal Cutoff value is the point where there is high true positive
#you can use the below code to get the values
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#ROC: receiver operating characterisics curve in logistic regressionare
#determing best cutoff/threshold value

import pylab as pl

i=np.arange(len(tpr)) #index of df
#here tpr is of 559 , so it will create a scale from 0 to 558
roc=pd.DataFrame({'fpr':pd.Series(fpr,index=i),
                  'tpr':pd.Series(tpr, index=i),
                  '1-fpr':pd.Series(1-fpr,index=i),
                  'tf':pd.Series(tpr-(1-fpr),index=i),
                  'thresholds':pd.Series(thresholds,index=i)
                  })
#we want to create a dataframe which comprises of columns fpr,tpr,1-fpr,tpr-(1-fpr),thresholds
#The optimal cutoff  would be where tpr is high and fpr is low
# tpr-(1-fpr) is zer or near to zero is the optimal cutoff point

#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc=auc(fpr,tpr)
print('Area under the curve: %f'%roc_auc)
#area is 0.7601

#tpr vs 1-fpr
#plot tpr vs 1-fpr
fig,ax=pl.subplot()
pl.plot(roc['tpr'],color='red')
pl.plot(roc['1-fpr'],color='blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver  Operating Characterisic')
ax.set_xticklabels([])
#The optimal cutoff point is one where tpr is high n fpr is low
#The optimal cutoff point is 0.317628
#so anything above this can be labeled as 0 else 1
#you can see from output that where TPR is crossong 1-FPR
#FPR is 36% and TPR is nearest to zero

#filling all the cells with zeroes
c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_threshold,'pred']=1
#let us check the classification report 
classification=classification_report(c1['pred'],c1['ATTORNEY'])
classification

#splitting the data into train and test
train_data,test_data=train_test_split(c1,test_size=0.3)
#Model Building
model=sm.logit('ATTORNEY ~ CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=train_data).fit()
model.summary()
#p values are below the condition is 0.05
#but SEATBELT has got statistically insignificant
model.summary2()
#AIC value is 1110.3782,AICscore are useful in comparison with other
#lower the AIC score better the model
#let us go for predictions
test_pred=logit_model.predict(test_data)
#creating new columnfor storing pedictedclass of ATTORNEY
test_data['test_pred']=np.zeros(402)
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1

#Confusion Matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
accuracy_test=(143+151)/(402)#Add current Values
accuracy_test

#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["ATTORNEY"])
classification_test

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["ATTORNEY"],test_pred)

#plot ROC Curve
plt.plot(fpr,tpr);plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")

#AUC
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

#prediction on train data
train_pred=logit_model.predict(train_data)
#Creating new column for storing predicted class of ATTORNEY
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix
accuracy_train=(315+347)/(938)
accuracy_train
#0.072174, this is going to  change with everytime when you 


#####################################################################
#Classifcation report
classification_train=classification_report(train_data["train_pred"],
                                           train_data["ATTORNEY"])
classification_train
#Accuracy=0.69

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(train_data["ATTORNEY"],train_pred)

#plotROC Curve
plt.plot(fpr,tpr);plt.xlabel("False Positive Rate");plt.ylabel("True Psitive Rate")

#AUC
roc_auc_train=metrics.auc(fpr,tpr)
roc_auc_train