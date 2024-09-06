# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:04:03 2024

@author: Gaurav Bombale
"""

"""
A logistics company recorded the time taken for delivery and the time taken 
for the sorting of the items for delivery. Build a Simple Linear Regression model
 to find the relationship between delivery time and sorting time with 
 delivery time as the target variable.
 Apply necessary transformations and record the RMSE and 
 correlation coefficient values for different models.
 
1.	Business Problem
1.1.	What is the business objective?
  1.1.1 —Delivery time is a crucial measure in transportation.
     Accurate delivery time prediction is also fundamental for operation and advanced logistic systems.
  1.1.2 prediction of delivery time with the increase of demand volume
        and e-commerce activities, logistic services face more challenges in order to maintain
         efficiency and customer retention
1.2.	Are there any constraints?
       Reliable long-term predictions remain challenging.There are unseen factors 
       which contribute in delibery time.

"""
##Database
#Delivery Time    float64
#Sorting Time       int64
import pandas as pd
import numpy as np
import seaborn as sns
logistic=pd.read_csv("delivery_time.csv")
logistic.dtypes
logistic.columns="del_time","sort_time"
#EDA

logistic.describe()
#Average delivery time is 16.71 and min is 8 and max is 29
#Average sort_time is 6.1 and min is 2.0 and max is 10
import matplotlib.pyplot as plt

sns.distplot(logistic.del_time)
#Data is normal but right skewed
plt.boxplot(logistic.del_time)
#No outliers but slight right skewed

sns.distplot(logistic.sort_time)
#Data is normal distributed 
plt.boxplot(logistic.sort_time)
#No outliers 
################
#Bivariant analysis
sns.regplot(x=logistic['sort_time'],y=logistic['del_time'])

#Data is linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=logistic['sort_time'],y=logistic['del_time'])
#The correlation coeficient is 0.8259<0.85 hence the correlation is moderate
#Let us check the direction of correlation
cov_output=np.cov(logistic.sort_time,y=logistic.del_time)[0,1]
cov_output
#10.65,it is positive means correlation will be positive
#########################################
# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('del_time~sort_time',data=logistic).fit()
#x=logistic['sort_time'],y=logistic['del_time']
model.summary()
#R-sqarred=0.682<0.85,model is not fit as good
#p=0.01<0.05 hence acceptable
#bita-0=6.5827
#bita-1=1.6490
pred1=model.predict(pd.DataFrame(logistic.sort_time))
pred1
################
#Regression line
sns.regplot(x=logistic['sort_time'],y=logistic['del_time'])
#plt.scatter(cal.cal_consumed,cal.wt_gained)
plt.plot(logistic.sort_time,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()
#################
##error calculations
res1=logistic.del_time-pred1
np.mean(res1)
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#2.79165
#
#################################
#let us try another model
#x=logistic['sort_time'],y=logistic['del_time
#x=log(logistic['sort_time'])
plt.scatter(x=np.log(logistic.sort_time),y=logistic.del_time)
#Data is not linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(logistic.sort_time),y=logistic.del_time)
#The correlation coeficient is 0.833<0.85 hence the correlation moderate
#r=0.8217
model2=smf.ols('logistic.del_time~np.log(logistic.sort_time)',data=logistic).fit()
#Y is wt_gained and X =log(cal_consumed)
model2.summary()
#R-sqarred=0.695<0.85,there is scope of improvement
#p=0.642>0.05 hence not acceptable
#bita-0=-6955.6501
#bita-1=np.log(cal_consumed)   948.37
pred2=model.predict(pd.DataFrame(logistic.sort_time))
pred2
################
#Regression line
plt.scatter(x=np.log(logistic.sort_time),y=logistic.del_time)

plt.plot(np.log(logistic.sort_time),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()
#################
##error calculations
res2=logistic.del_time-pred2
np.mean(res2)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#2.7916
#This model has very poor r value,R-squarred value 
#Hence let us try another model
##########################################
#Now let us make logY and X as is
#x=logistic['sort_time'],y=np.log(logistic['del_time'])
plt.scatter(x=(logistic.sort_time),y=np.log(logistic.del_time))
#Data is not linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=(logistic.sort_time),y=np.log(logistic.del_time))
#The correlation coeficient is 0.8431<0.85 hence the correlation is moderate
#r=0.8431
model3=smf.ols('np.log(del_time)~sort_time',data=logistic).fit()
#Y is log(AT) and X =Waist
model3.summary()
#R-sqarred=0.711<0.85
#p=0.000<0.05 hence acceptable
#bita-0=2.1214   
#bita-1=   0.1056
pred3=model3.predict(pd.DataFrame(logistic.sort_time))
pred3_at=np.exp(pred3)
pred3_at
################
#Regression line
plt.scatter(logistic.sort_time,np.log(logistic.del_time))
plt.plot(logistic.sort_time,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res3=logistic.del_time-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#2.940,higher than earlier two models

#Hence let us try another model
#######################################
#Now let us make Y=np.log(cal.wt_gained) and X=cal.cal_consumed,X*X=cal.cal_consumed*cal.cal_consumed
x=logistic['sort_time']+logistic['sort_time']*logistic['sort_time'],y=np.log(logistic['del_time'])
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(del_time)~sort_time+I(sort_time*sort_time)',data=logistic).fit()
#Y=np.log(cal.wt_gained) and X=cal.cal_consumed
model4.summary()
#R-sqarred=0.765<0.85
#p=0.022 <0.05 hence acceptable
#bita-0=1.6997 
#bita-1=   0.2659 
pred4=model4.predict(pd.DataFrame(logistic.sort_time))
pred4
pred4_at=np.exp(pred4)
pred4_at
################
#Regression line
plt.scatter(logistic.sort_time,np.log(logistic.del_time))
plt.plot(logistic.sort_time,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res4=logistic.del_time-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#2.79
#Better as compared to third model but higher than second model,which was 103.30
#########################################
data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse
###################
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(logistic,test_size=0.3)

plt.scatter(train.sort_time,train.del_time)
plt.scatter(test.sort_time,test.del_time)


#Now let us check the correlation coeficient
np.corrcoef(x=train.sort_time,y=train.del_time)
#The correlation coeficient is 0.9274>0.85 hence the correlation is good
#r=0.9274
final_model=smf.ols('del_time~sort_time',data=logistic).fit()
#Y is del_timeand X =sort_time
final_model.summary()
#R-sqarred=0.682<0.85,there is scope of improvement
#p=0.01<0.05 hence acceptable
#bita-0= 6.5827 
#bita-1=   1.6490
test_pred=final_model.predict(pd.DataFrame(test))
test_pred

test_res=test.del_time-test_pred

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#3.6574
########################################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred

train_res=train.del_time-train_pred

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#2.236
####################################################


