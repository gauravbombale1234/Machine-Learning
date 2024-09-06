# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:04:51 2024

@author: Gaurav Bombale
"""

"""
A certain university wants to understand the 
relationship between students’ SAT scores and 
their GPA. Build a Simple Linear Regression model
 with GPA as the target variable and record the RMSE 
 and correlation coefficient values for different models.

1.	Business Problem
1.1.	What is the business objective?
  1.1.1 That said, keep in mind that your GPA is not at all 
worthless! While the SAT might have won the match, 
the GPA manages to steal several rounds. It is the 
only numerical data that can reveal hard work, 
self-discipline, and consistency. Together with your 
transcript, your GPA can show improvement over time 
and intellectual growth. These qualities are quite 
valuable when you are being compared to another student with a similar SAT score
 but lesser GPA. It is still important to take rigorous
 classes and earn good grades to bolster your
 transcript and GPA.

1.2.	Are there any constraints?
If you have high SAT scores, are you likely to 
get better grades in college?  Perhaps unsurprisingly,
 most research studies find that your SAT scores 
 do predict college success - to an extent. 
 The relationship isn't particularly strong, 
 which means that if you have high SAT scores,
 you're only slightly more likely to have higher college
 grades than a student who had lower scores. 
 Research shows a similar relationship between SAT scores
 and high school grades.        

"""
##Database
#SAT_Scores      int64
#GPA           float64             
import pandas as pd
import numpy as np
import seaborn as sns
sat=pd.read_csv("SAT_GPA.csv")
sat.dtypes

#EDA

sat.describe()
#Average Sat score is 491.81 and min is 202 and max is 797
#Average GPA is 2.84 and min is 2 and max is 3.9
import matplotlib.pyplot as plt


sns.distplot(sat.SAT_Scores)
#Data is normal 
plt.boxplot(sat.SAT_Scores)
#No outliers 

sns.distplot(sat.GPA)
#Data is normal distributed ,bimodal
plt.boxplot(sat.GPA)
#No outliers 
################
#Bivariant analysis
sns.regplot(x=sat.SAT_Scores,y=sat.GPA)

#Data is not linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=sat.SAT_Scores,y=sat.GPA)
#The correlation coeficient is 0.2935<0.85 hence the correlation is very poor
#Let us check the direction of correlation
cov_output=np.cov(sat.SAT_Scores,sat.GPA)[0,1]
cov_output
#27.7777it is positive means correlation will be positive
#########################################
# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('GPA~SAT_Scores',data=sat).fit()
#x=sat.SAT_Scores,y=sat.GPA
model.summary()
#R-sqarred=0.086<<0.85,model is not best fit 
#p=0.00<0.05 hence acceptable

pred1=model.predict(pd.DataFrame(sat.SAT_Scores))
pred1
################
#Regression line
sns.regplot(x=sat.SAT_Scores,y=sat.GPA)

plt.plot(sat.SAT_Scores,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()
#################
##error calculations
res1=sat.GPA-pred1
np.mean(res1)
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#0.5159
#
#################################
#let us try another model
#x=sat.SAT_Scores,y=sat.GPA

#x=log(sat.SAT_Scores)
#plt.scatter(x=np.log(hr2.yrs_exp),y=hr2.sal)
sns.regplot(x=np.log(sat.SAT_Scores),y=sat.GPA)
#Data is not at all linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(sat.SAT_Scores),y=sat.GPA)
#The correlation coeficient is 0.2777<<0.85 hence the correlation very poor
#r=0.2777
model2=smf.ols('GPA~np.log(SAT_Scores)',data=sat).fit()
#Y is GPA and X =log(SAT_Scores)
model2.summary()
#R-sqarred=0.077<<<0.85
#p=0.412 >0.05 hence not acceptable

pred2=model.predict(pd.DataFrame(sat.SAT_Scores))
pred2
################
#Regression line
plt.scatter(x=np.log(sat.SAT_Scores),y=sat.GPA)
plt.plot(np.log(sat.SAT_Scores),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()
#################
##error calculations
res2=sat.GPA-pred2
np.mean(res2)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#0.5159
#This model has same R-squarred value 
#Hence let us try another model
##########################################
#Now let us make logY and X as is
#x=sat.SAT_Scores,y=np.log(sat.GPA)

plt.scatter(x=(sat.SAT_Scores),y=np.log(sat.GPA))
#Data is  linearly scattered,direction positive,strength:good
#Now let us check the correlation coeficient
np.corrcoef(x=(sat.SAT_Scores),y=np.log(sat.GPA))
#The correlation coeficient is 0.2940<0.85 hence the correlation is very poor
#r=0.2940
model3=smf.ols('np.log(GPA)~SAT_Scores',data=sat).fit()

model3.summary()
#R-sqarred=0.086<<0.85
#p=0.000<0.05 hence acceptable

pred3=model3.predict(pd.DataFrame(sat.SAT_Scores))
pred3_at=np.exp(pred3)
pred3_at
################
#Regression line
plt.scatter(sat.SAT_Scores,np.log(sat.GPA))
plt.plot(sat.SAT_Scores,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res3=sat.GPA-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#0.5175 

#Hence let us try another model
#######################################
#Now let us make Y=np.log(sat.GPA) and X=sat.SAT_Scores,X*X=sat.SAT_Scores*sat.SAT_Scores
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(GPA)~SAT_Scores+I(SAT_Scores*SAT_Scores)',data=sat).fit()

model4.summary()
#R-sqarred=0.094<<0.85
#p=0.00<0.05 hence acceptable

#I(yrs_exp * yrs_exp)    -0.0066
pred4=model4.predict(pd.DataFrame(sat.SAT_Scores))
pred4
pred4_at=np.exp(pred4)
pred4_at
################
#Regression line
plt.scatter(sat.SAT_Scores,np.log(sat.GPA))
plt.plot(sat.SAT_Scores,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
#################
##error calculations
res4=sat.GPA-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#0.5144  
#This is the best model

#########################################
data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse
###################
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(sat,test_size=0.2)

plt.scatter(train.SAT_Scores,train.GPA)
plt.scatter(test.SAT_Scores,test.GPA)


#Now let us check the correlation coeficient
np.corrcoef(train.SAT_Scores,train.GPA)
#The correlation coeficient is 0.27433<<0.85 hence the correlation is very poor
#r=0.27433

final_model=smf.ols('np.log(GPA)~SAT_Scores+I(SAT_Scores*SAT_Scores)',data=sat).fit()

final_model.summary()
#R-sqarred=0.094<0.85
#p=0.00<0.05 hence acceptable

##########################################
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

test_res=test.GPA-test_pred_at

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#0.5053
################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at

train_res=train.GPA-train_pred_at

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#0.5167
