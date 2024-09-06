# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:04:03 2024

@author: Gaurav Bombale
"""

"""
A certain food-based company conducted a survey with the help
 of a fitness company to find the relationship between 
 a person’s weight gain and the number of calories 
 they consumed in order to come up with diet plans
 for these individuals. Build a Simple Linear Regression model 
 with calories consumed as the target variable.
 Apply necessary transformations and record the RMSE and correlation coefficient values for different models. 

1. Business Problem
1.1.	What is the business objective?
  1.1.1 If people consumed only the number of calories 
        needed every day, they would probably have healthy lives.
        Calorie consumption that is too low or too high will 
        eventually lead to health problems.
  1.1.2 prediction of The number of calories in food tells 
      us how much potential energy they contain. 
      It is not only calories that are important, 
      but also the substance from which the calories are taken.
1.2.	Are there any constraints?

       It also appears that weight regain occurs regardless 
       of the type of diet used for weight loss, although some 
       diets are linked to less regain than others.

"""
##Database
#Weight gained (grams)    int64
#Calories Consumed        int64
import pandas as pd
import numpy as np
import seaborn as sns
cal=pd.read_csv("calories_consumed.csv")
cal.dtypes
cal.columns="wt_gained","cal_consumed"
#EDA

cal.describe()
#Average weight gained is 357.71 and min is 62.00 and max is 1100 gms
#Average Calories consumed is 2340 and min is 1400 and max is 3900
import matplotlib.pyplot as plt
plt.bar(height=cal.wt_gained,x=np.arange(1,110,1))
sns.distplot(cal.wt_gained)
#Data is normal but right skewed
plt.boxplot(cal.wt_gained)
#No outliers but right skewed
plt.bar(height=cal.cal_consumed,x=np.arange(1,110,1))
sns.distplot(cal.cal_consumed)
#Data is normal distributed 
plt.boxplot(cal.cal_consumed)
#No outliers but slight right skewed
################
#Bivariant analysis
plt.scatter(x=cal.cal_consumed,y=cal.wt_gained)
#Data is linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=cal.cal_consumed,y=cal.wt_gained)
#The correlation coeficient is 0.94699101>0.85 hence the correlation is strong
#Let us check the direction of correlation
cov_output=np.cov(cal.cal_consumed,cal.wt_gained)[0,1]
cov_output
#237669.45,it is positive means correlation will be positive
#########################################
# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('wt_gained~cal_consumed',data=cal).fit()
#Y is AT and X is waist
model.summary()
#R-sqarred=0.897>0.85,model is good
#p=00<0.05 hence acceptable
#bita-0=-625.7524 
#bita-1=0.4202
pred1=model.predict(pd.DataFrame(cal.cal_consumed))
pred1
################
#Regression line
plt.scatter(cal.cal_consumed,cal.wt_gained)
plt.plot(cal.cal_consumed,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()
#################
##error calculations
res1=cal.wt_gained-pred1
np.mean(res1)
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#2027.04
#RMSE is very high
#################################
#let us try another model
#x=log(cal.cal_consumed)
plt.scatter(x=np.log(cal.cal_consumed),y=cal.wt_gained)
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(cal.cal_consumed),y=cal.wt_gained)
#The correlation coeficient is 0.89872528>0.85 hence the correlation is good
#r=0.8217
model2=smf.ols('wt_gained~np.log(cal_consumed)',data=cal).fit()
#Y is wt_gained and X =log(cal_consumed)
model2.summary()
#R-sqarred=0.808<0.85,there is scope of improvement
#p=00<0.05 hence acceptable
#bita-0=-6955.6501
#bita-1=np.log(cal_consumed)   948.37
pred2=model.predict(pd.DataFrame(cal.cal_consumed))
pred2
################
#Regression line
plt.scatter(np.log(cal.cal_consumed),cal.wt_gained)
plt.plot(np.log(cal.cal_consumed),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()
#################
##error calculations
res2=cal.wt_gained-pred2
np.mean(res2)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#103.30
#Better as compared to earlier,which was 2027.04
#Hence let us try another model
##########################################
#Now let us make logY and X as is
plt.scatter(x=(cal.cal_consumed),y=np.log(cal.wt_gained))
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=(cal.cal_consumed),y=np.log(cal.wt_gained))
#The correlation coeficient is 0.8185<0.85 hence the correlation is moderate
#r=0.9368
model3=smf.ols('np.log(wt_gained)~cal_consumed',data=cal).fit()
#Y is log(AT) and X =Waist
model3.summary()
#R-sqarred=0.878>0.85
#p=0.000<0.05 hence acceptable
#bita-0=2.8387   
#bita-1=   0.0011
pred3=model3.predict(pd.DataFrame(cal.cal_consumed))
pred3_at=np.exp(pred3)
pred3_at
################
#Regression line
plt.scatter(cal.cal_consumed,np.log(cal.wt_gained))
plt.plot(cal.cal_consumed,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res3=cal.wt_gained-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#118.045
# Better as compared to first model but higher than second model,which was 103.30
#Hence let us try another model
#######################################
#Now let us make Y=np.log(cal.wt_gained) and X=cal.cal_consumed,X*X=cal.cal_consumed*cal.cal_consumed
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(wt_gained)~cal_consumed+I(cal_consumed*cal_consumed)',data=cal).fit()
#Y=np.log(cal.wt_gained) and X=cal.cal_consumed
model4.summary()
#R-sqarred=0.0.878>0.85
#p=0.022 <0.05 hence acceptable
#bita-0=2.8287
#bita-1=   0.0011 
pred4=model4.predict(pd.DataFrame(cal.cal_consumed))
pred4
pred4_at=np.exp(pred4)
pred4_at
################
#Regression line
plt.scatter(cal.cal_consumed,np.log(cal.wt_gained))
plt.plot(cal.cal_consumed,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res4=cal.wt_gained-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#117.41
#Better as compared to third model but higher than second model,which was 103.30
#########################################
data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse
###################
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(cal,test_size=0.3)

plt.scatter(np.log(train.cal_consumed),train.wt_gained)
plt.scatter(np.log(test.cal_consumed),test.wt_gained)

#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(train.cal_consumed),y=train.wt_gained)
#The correlation coeficient is 0.9224386>0.85 hence the correlation is good
#r=0.9224386
final_model=smf.ols('wt_gained~np.log(cal_consumed)',data=cal).fit()
#Y is wt_gained and X =log(cal_consumed)
final_model.summary()
#R-sqarred=0.808<0.85,there is scope of improvement
#p=00<0.05 hence acceptable
#bita-0=-6955.6501
#bita-1=np.log(cal_consumed)   948.37
test_pred=final_model.predict(pd.DataFrame(test))
test_pred

test_res=test.wt_gained-test_pred

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#111.23
########################################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred

train_res=train.wt_gained-train_pred

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#155.094
####################################################
