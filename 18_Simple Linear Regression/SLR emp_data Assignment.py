# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:04:03 2024

@author: Gaurav Bombale
"""

"""
A certain organization wants an early estimate of their employee churn out
 rate. So the HR department gathered the data regarding the employee’s
 salary hike and the churn out rate in a financial year. 
 The analytics team will have to perform an analysis and predict 
 an estimate of employee churn based on the salary hike. 
 Build a Simple Linear Regression model with churn out rate as the 
 target variable. Apply necessary transformations and record the RMSE 
 and correlation coefficient values for different models.
 
1.	Business Problem
1.1.	What is the business objective?
  1.1.1 In Research, it was found that employee churn will be affected 
  by age, tenure, pay, job satisfaction, salary, working conditions,
  growth potential and employee’s perceptions of fairness. 
  Some other variables such as age, gender, ethnicity, education, 
  and marital status, were essential factors in the prediction of employee
  churn.
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
#Salary_hike       int64
#Churn_out_rate    int64
import pandas as pd
import numpy as np
import seaborn as sns
hr=pd.read_csv("emp_data.csv")
hr.dtypes

#EDA

hr.describe()
#Average salary hike is 1688.60 and min is 1580 and max is 1870
#Average churn_out_rate is 72.90 and min is 60 and max is 92
import matplotlib.pyplot as plt

sns.distplot(hr.Salary_hike)
#Data is normal 
plt.boxplot(hr.Salary_hike)
#No outliers but slight right skewed

sns.distplot(hr.Churn_out_rate)
#Data is normal distributed 
plt.boxplot(hr.Churn_out_rate)
#No outliers 
################
#Bivariant analysis
sns.regplot(x=hr.Salary_hike,y=hr.Churn_out_rate)

#Data is linearly scattered,direction nagative,
#Now let us check the correlation coeficient
np.corrcoef(x=hr.Salary_hike,y=hr.Churn_out_rate)
#The correlation coeficient is -0.9117>-0.85 hence the correlation is strong
#Let us check the direction of correlation
cov_output=np.cov(hr.Salary_hike,hr.Churn_out_rate)[0,1]
cov_output
#-861.266,it is negative means correlation will be negative
#########################################
# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('Churn_out_rate~Salary_hike',data=hr).fit()
#x=logistic['sort_time'],y=logistic['del_time']
model.summary()
#R-sqarred=0.831<0.85,model is not fit as good
#p=0.01<0.05 hence acceptable
#bita-0=244.3649
#bita-1=-0.1015
pred1=model.predict(pd.DataFrame(hr.Salary_hike))
pred1
################
#Regression line
sns.regplot(hr.Salary_hike,hr.Churn_out_rate)
#plt.scatter(cal.cal_consumed,cal.wt_gained)
plt.plot(hr.Salary_hike,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()
#################
##error calculations
res1=hr.Churn_out_rate-pred1
np.mean(res1)
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#3.9975
#
#################################
#let us try another model
#x=hr.Salary_hike,y=hr.Churn_out_rate
#x=log(hr.Salary_hike)
plt.scatter(x=np.log(hr.Salary_hike),y=hr.Churn_out_rate)
#Data is not linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(hr.Salary_hike),y=hr.Churn_out_rate)
#The correlation coeficient is -0.9212>-0.85 hence the correlation moderate
#r=-0.9212
model2=smf.ols('Churn_out_rate~np.log(Salary_hike)',data=hr).fit()
#Y is Churn_out_rate and X =log(Salary_hike)
model2.summary()
#R-sqarred=0.849<0.85
#p=0.00>0.05 hence not acceptable
#bita-0=1381.45
#bita-1=np.log(Salary_hike)  -176.1097 
pred2=model.predict(pd.DataFrame(hr.Salary_hike))
pred2
################
#Regression line
plt.scatter(x=np.log(hr.Salary_hike),y=hr.Churn_out_rate)

plt.plot(np.log(hr.Salary_hike),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()
#################
##error calculations
res2=hr.Churn_out_rate-pred2
np.mean(res2)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#3.99
#This model has same R-squarred value 
#Hence let us try another model
##########################################
#Now let us make logY and X as is
#x=hr.Salary_hike,y=hr.Churn_out_rate
plt.scatter(x=(hr.Salary_hike),y=np.log(hr.Churn_out_rate))
#Data is not linearly scattered,direction negative,strength:good
#Now let us check the correlation coeficient
np.corrcoef(x=(hr.Salary_hike),y=np.log(hr.Churn_out_rate))
#The correlation coeficient is -0.9346>-0.85 hence the correlation is moderate
#r=-0.9346
model3=smf.ols('np.log(Churn_out_rate)~Salary_hike',data=hr).fit()
#Y is log(AT) and X =Waist
model3.summary()
#R-sqarred=0.874>0.85
#p=0.000<0.05 hence acceptable
#bita-0=6.6383    
#bita-1=   -0.0014
pred3=model3.predict(pd.DataFrame(hr.Salary_hike))
pred3_at=np.exp(pred3)
pred3_at
################
#Regression line
plt.scatter(hr.Salary_hike,np.log(hr.Churn_out_rate))
plt.plot(hr.Salary_hike,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res3=hr.Churn_out_rate-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#3.541 better than earlier model

#Hence let us try another model
#######################################
#Now let us make Y=np.log(cal.wt_gained) and X=cal.cal_consumed,X*X=cal.cal_consumed*cal.cal_consumed
y=np.log(hr.Churn_out_rate),(hr.Salary_hike)+(hr.Salary_hike)*(hr.Salary_hike)
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(Churn_out_rate)~Salary_hike+I(Salary_hike*Salary_hike)',data=hr).fit()
#Y=np.log(cal.wt_gained) and X=cal.cal_consumed
model4.summary()
#R-sqarred=0.984>0.85
#p=0.00<0.05 hence acceptable
#bita-0=28.887
#bita-1=   -0.014
pred4=model4.predict(pd.DataFrame(hr.Salary_hike))
pred4
pred4_at=np.exp(pred4)
pred4_at
################
#Regression line
plt.scatter(hr.Salary_hike,np.log(hr.Churn_out_rate))
plt.plot(hr.Salary_hike,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res4=hr.Churn_out_rate-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#1.32678 
#This is the best model

#########################################
data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse
###################
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(hr,test_size=0.3)

plt.scatter(train.Salary_hike,train.Churn_out_rate)
plt.scatter(test.Salary_hike,test.Churn_out_rate)


#Now let us check the correlation coeficient
np.corrcoef(train.Salary_hike,train.Churn_out_rate)
#The correlation coeficient is -0.9559>0.85 hence the correlation is good
#r=-0.9559

final_model=smf.ols('np.log(Churn_out_rate)~Salary_hike+I(Salary_hike*Salary_hike)',data=hr).fit()

final_model.summary()
#R-sqarred=0.984>0.85
#p=0.00<0.05 hence acceptable
#bita-0=28.887
#bita-1=   -0.014
##########################################
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

test_res=test.Churn_out_rate-test_pred_at

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#0.76106
################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at

train_res=train.Churn_out_rate-train_pred_at

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#1.5055
