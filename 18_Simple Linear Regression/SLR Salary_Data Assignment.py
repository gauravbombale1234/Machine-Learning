# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:04:50 2024

@author: Gaurav Bombale
"""

"""
The head of HR of a certain organization wants to automate 
their salary hike estimation. The organization consulted 
an analytics service provider and asked them to build a basic
 prediction model by providing them with a dataset that 
 contains the data about the number of years of experience 
 and the salary hike given accordingly. Build a Simple Linear
 Regression model with salary as the target variable. 
 Apply necessary transformations and record the RMSE 
 and correlation coefficient values for different models.

1.	Business Problem
1.1.	What is the business objective?
  1.1.1 Employees keep switching the company to get the expected
salary. And it leads to loss of the company and to overcome this
loss algorith that derives what if the employee gets the
desired/expected salary from the Company or Organization. In
this Competitive world everyone has a higher expectation and
goals. But we cannot randomly provide everyone their expected
salary there should be a system which should measure the
ability of the Employee for the Expected salary.

  1.1.2 Organizations can tackle this problem by applying machine learning 
  techniques to predict employee churn, which helps them in 
  taking necessary actions.
1.2.	Are there any constraints?
Accurate  predictions  enable organizations to take action for retention 
or succession planning of  employees. 
 However,  the  data  for  this  modeling  problem comes from HR 
 Information Systems (HRIS); these are typically under-funded  compared 
 to  the  Information  Systems  of  other domains  in  the  organization 
 which  are  directly  related  to  its priorities. 
 This leads to the prevalence  of noise  in the  data that renders 
 predictive  models  prone  to  over-fitting  and  hence inaccurate.  
 This  is  the  key  challenge

"""
##Database
#YearsExperience    float64
#Salary             float64
import pandas as pd
import numpy as np
import seaborn as sns
hr2=pd.read_csv("Salary_Data.csv")
hr2.dtypes

#EDA

hr2.describe()
#Average YearsExperience is 5.31 and min is 1.10 and max is 10.50
#Average salary is 76003 and min is 37731 and max is 122391
import matplotlib.pyplot as plt
hr2.columns="yrs_exp","sal"
hr2.dtypes
sns.distplot(hr2.yrs_exp)
#Data is normal 
plt.boxplot(hr2.yrs_exp)
#No outliers but slight right skewed

sns.distplot(hr2.sal)
#Data is normal distributed ,bimodal
plt.boxplot(hr2.sal)
#No outliers 
################
#Bivariant analysis
sns.regplot(x=hr2.yrs_exp,y=hr2.sal)

#Data is linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=hr2.yrs_exp,y=hr2.sal)
#The correlation coeficient is 0.9782>0.85 hence the correlation is strong
#Let us check the direction of correlation
cov_output=np.cov(hr2.yrs_exp,hr2.sal)[0,1]
cov_output
#76106.30it is positive means correlation will be positive
#########################################
# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('sal~yrs_exp',data=hr2).fit()
#x=hr2.yrs_exp,y=hr2.sal
model.summary()
#R-sqarred=0.957>0.85,model is best fit 
#p=0.00<0.05 hence acceptable
#bita-0=2.579e+04
#bita-1=9449.9623
pred1=model.predict(pd.DataFrame(hr2.yrs_exp))
pred1
################
#Regression line
sns.regplot(hr2.yrs_exp,hr2.sal)

plt.plot(hr2.yrs_exp,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()
#################
##error calculations
res1=hr2.sal-pred1
np.mean(res1)
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#5592.04
#
#################################
#let us try another model
#x=hr2.yrs_exp,y=hr2.sal
#x=log(hr2.yrs_exp)
#plt.scatter(x=np.log(hr2.yrs_exp),y=hr2.sal)
sns.regplot(x=np.log(hr2.yrs_exp),y=hr2.sal)
#Data is  linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(hr2.yrs_exp),y=hr2.sal)
#The correlation coeficient is 0.9240>0.85 hence the correlation strong
#r=0.9240
model2=smf.ols('sal~np.log(yrs_exp)',data=hr2).fit()
#Y is sal and X =log(yrs_exp)
model2.summary()
#R-sqarred=0.854=0.85
#p=0.007>0.05 hence not acceptable
#bita-0=1.493e+04
#bita-1=np.log(yrs_exp)  4.058e+04 
pred2=model.predict(pd.DataFrame(hr2.yrs_exp))
pred2
################
#Regression line
plt.scatter(x=np.log(hr2.yrs_exp),y=hr2.sal)
plt.plot(np.log(hr2.yrs_exp),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()
#################
##error calculations
res2=hr2.sal-pred2
np.mean(res2)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#5592.04
#This model has same R-squarred value 
#Hence let us try another model
##########################################
#Now let us make logY and X as is
#x=x=(hr2.yrs_exp),y=np.log(hr2.sal
plt.scatter(x=(hr2.yrs_exp),y=np.log(hr2.sal))
#Data is  linearly scattered,direction positive,strength:good
#Now let us check the correlation coeficient
np.corrcoef(x=(hr2.yrs_exp),y=np.log(hr2.sal))
#The correlation coeficient is 0.9653>0.85 hence the correlation is strong
#r=0.9653
model3=smf.ols('np.log(sal)~yrs_exp',data=hr2).fit()
#Y is log(sal) and X =yrs_exp
model3.summary()
#R-sqarred=0.932>0.85
#p=0.000<0.05 hence acceptable
#bita-0=10.5074   
#bita-1=   0.1255 
pred3=model3.predict(pd.DataFrame(hr2.yrs_exp))
pred3_at=np.exp(pred3)
pred3_at
################
#Regression line
plt.scatter(hr2.yrs_exp,np.log(hr2.sal))
plt.plot(hr2.yrs_exp,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res3=hr2.sal-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#7213.23 higher than earlier model

#Hence let us try another model
#######################################
#Now let us make Y=np.log(hr2.sal) and X=hr2.yrs_exp,X*X=hr2.yrs_exp*hr2.yrs_exp
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(sal)~yrs_exp+I(yrs_exp*yrs_exp)',data=hr2).fit()

model4.summary()
#R-sqarred=0.949>0.85
#p=0.00<0.05 hence acceptable
#bita-0=-1.5369
#bita-1=   0.2024 
#I(yrs_exp * yrs_exp)    -0.0066
pred4=model4.predict(pd.DataFrame(hr2.yrs_exp))
pred4
pred4_at=np.exp(pred4)
pred4_at
################
#Regression line
plt.scatter(hr2.yrs_exp,np.log(hr2.sal))
plt.plot(hr2.yrs_exp,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res4=hr2.sal-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#5391.08  #This is the lowest
#This is the best model

#########################################
data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse
###################
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(hr2,test_size=0.2)

plt.scatter(train.yrs_exp,train.sal)
plt.scatter(test.yrs_exp,test.sal)


#Now let us check the correlation coeficient
np.corrcoef(train.yrs_exp,train.sal)
#The correlation coeficient is 0.9816>0.85 hence the correlation is strong
#r=0.9816

final_model=smf.ols('np.log(sal)~yrs_exp+I(yrs_exp*yrs_exp)',data=hr2).fit()

final_model.summary()
#R-sqarred=0.949>0.85
#p=0.00<0.05 hence acceptable

##########################################
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

test_res=test.sal-test_pred_at

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#3475.69
################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at

train_res=train.sal-train_pred_at

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#5771.44
