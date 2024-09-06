# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:54:41 2024

@author: Gaurav Bombale
"""
"""
1.	The dataset consists of monthly totals of international airline passengers 
from 1995 to 2002. Our main aim is to predict the number of passengers for the 
next five years using time series forecasting. Prepare a document for each model 
explaining how many dummy variables you have created and also include the RMSE 
value for each model.
"""
import pandas as pd
airlines=pd.read_excel("Airlines Data.xlsx")

airlines.dtypes
#To extract the years
airlines["Month"].dt.year
#To extract the month
airlines["Month"].dt.month
airlines["Year"]=airlines["Month"].dt.year
airlines["Months"]=airlines["Month"].dt.month
airlines.drop(["Month"],axis=1,inplace=True)

#####

airlines['Months']=pd.to_datetime(airlines['Months'],format='%m').dt.month_name().str.slice(stop=3)
airlines=airlines[['Months','Passengers']] 
#airlines["period"] = airlines["Months"].astype(str) + airlines["Year"].astype(str)        


a=airlines["Months"][0]
a[0:3]

#a=airlines["period"][0]
#a[5:7]
#airlines['month']=0
#airlines['year']=0
for i in range(96):
    a=airlines["Months"][i]
    airlines["Months"][i]=a[0:3]

#for i in range(96):
 #   a=airlines["period"][i]
  #  airlines["Year"][i]=a[5:7]
    
#airlines["period"] = airlines["Months"].astype(str) + airlines["Year"].astype(str)        
#airlines.drop(["Months"],axis=1,inplace=True)
#airlines.drop(["Year"],axis=1,inplace=True)

#airlines=airlines[['period','Passengers']]
#########
    #for all these months create dummy variables
Months_dummies=pd.DataFrame(pd.get_dummies(airlines['Months']))
#month_dummies=pd.DataFrame(pd.get_dummies(airlines['Months']))
## now let us concatenate these dummy values to dataframe
airlines1=pd.concat([airlines,Months_dummies],axis=1)
#airlines2=pd.concat([airlines1,month_dummies],axis=1)
# you can check the dataframe airlines1

# similarly we need to create column t
import numpy as np
airlines1['t']=np.arange(1,97)
airlines1['t_squared']=airlines1['t']*airlines1['t']
airlines1['log_Passengers']=np.log(airlines1['Passengers'])

#Now let us check the visuals of the passengers
airlines1.Passengers.plot()
#You will get increasing trend with gradual  increasing
# we have to forecast Passengers in next 12 months,hence horizon=12,even 
#season=12,so validating data will be 12 and training will 96-12=84
Train=airlines1.head(84)
Test=airlines1.tail(12)
# Now let us apply linear regression
import statsmodels.formula.api as smf
##Linear model
linear_model=smf.ols("Passengers~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))

rmse_linear
##Exponential model
Exp_model=smf.ols("log_Passengers~t",data=Train).fit()
pred_Exp=pd.Series(Exp_model.predict(pd.DataFrame(Test['t'])))
rmse_Exp=np.sqrt(np.mean((np.array(Test['Passengers'])-np.exp(pred_Exp))**2))

rmse_Exp

##Quadratic model
Quad=smf.ols("Passengers~t+t_squared",data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_squared"]]))
rmse_Quad=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))

rmse_Quad
################### Additive seasonality ########################
add_sea=smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=Train).fit()


pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
#pred_add_sea = pd.Series(add_sea.predict(Test[['Jan49','Feb49','Mar49','Apr49','May49','Jun49','Jul49','Aug49','Sep49','Oct49','Nov49','Dec49','Jan50','Feb50','Mar50','Apr50','May50','Jun50','Jul50','Aug50','Sep50','Oct50','Nov50','Dec50','Jan51','Feb51','Mar51','Apr51','May51','Jun51','Jul51','Aug51','Sep51','Oct51','Nov51','Dec51','Jan52','Feb52','Mar52','Apr52','May52','Jun52','Jul52','Aug52','Sep52','Oct52','Nov52','Dec52','Jan53','Feb53','Mar53','Apr53','May53','Jun53','Jul53','Aug53','Sep53','Oct53','Nov53']]))

rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(pred_add_sea))**2))
rmse_add_sea
##Multiplicative seasonability model
mul_sea=smf.ols("log_footfalls~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul_sea = pd.Series(mul_sea.predict(Test))
rmse_mul_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_mul_sea)))**2))

rmse_mul_sea

################### Additive seasonality with quadratic trend ########################

add_sea_quad = smf.ols('Footfalls ~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
##Multiplicative seasonability linear model
mul_add_sea=smf.ols("log_footfalls~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_mul_add_sea)))**2))

rmse_mul_add_sea
###let us create a dataframe and add all these rmse_values
data={"Model":pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
data
## Now let us test the model with full data
predict_data = pd.read_excel("C:/360DG/Datasets/Predict_new.xlsx")

model_full = smf.ols('Footfalls ~ t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=walmart1).fit()

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Footfalls"] = pd.Series(pred_new)

#########AR model
## let us find out the errors
full_res=walmart1.Footfalls-model_full.predict(walmart1)
#Actual value=walmart1.Footfalls and PV=model_full.predict(walmart1)
#Now plot ACF plot on residue/error
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(full_res,lags=12)
# if we will observe the plot r0 is highest but it is correlation with
#itself hence ignored second highest is r1 hence lags-1 is considered
#PACF is partial auto correlation function
# it finds correlation of present with lags of residuals of time series
tsa_plots.plot_pacf(full_res,lags=12)
from statsmodels.tsa.ar_model import AutoReg
model_ar=AutoReg(full_res,lags=[1])
model_fit=model_ar.fit()
print("coeficients %s" % model_fit.params)
# again calculate residuals for new model
pred_res=model_fit.predict(start=len(full_res), end=len(full_res)+len(predict_data)-1,dynamic=False)


pred_res.reset_index(drop=True,inplace=True)
final_pred=pred_new+pred_res
final_pred
