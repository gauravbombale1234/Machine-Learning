# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:54:42 2024

@author: Gaurav Bombale
"""
"""
3. A plastics manufacturing plant has recorded their monthly sales data from 
1949 to 1953. Perform forecasting on the data and bring out insights from 
it and forecast the sale for the next year. 
"""
import pandas as pd
plastic=pd.read_csv("PlasticSales.csv")

plastic.dtypes

#####
p=plastic["Month"][0]
p[0:3]

# before we will extract ,let us create new column called months to store extracted values
plastic['month']=0


for i in range(60):
    p=plastic["Month"][i]
    plastic["month"][i]=p[0:3]
    #for all these months create dummy variables
month_dummies=pd.DataFrame(pd.get_dummies(plastic['month']))
#month_dummies=pd.DataFrame(pd.get_dummies(airlines['Months']))
## now let us concatenate these dummy values to dataframe
plastic1=pd.concat([plastic,month_dummies],axis=1)
#airlines2=pd.concat([airlines1,month_dummies],axis=1)
# you can check the dataframe plastic1

# similarly we need to create column t
import numpy as np
plastic1['t']=np.arange(1,61)
plastic1['t_squared']=plastic1['t']*plastic1['t']
plastic1['log_Sales']=np.log(plastic1['Sales'])

#Now let us check the visuals of the passengers
plastic1.Sales.plot()
#You will get  trend with gradual  increasing and linear
# we have to forecast Sales in next 1 year,hence horizon=12,even 
#season=12,so validating data will be 12 and training will 60-12=48
Train=plastic1.head(48)
Test=plastic1.tail(12)
# Now let us apply linear regression
import statsmodels.formula.api as smf
##Linear model
linear_model=smf.ols("Sales~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))

rmse_linear
##Exponential model
Exp_model=smf.ols("log_Sales~t",data=Train).fit()
pred_Exp=pd.Series(Exp_model.predict(pd.DataFrame(Test['t'])))
rmse_Exp=np.sqrt(np.mean((np.array(Test['Sales'])-np.exp(pred_Exp))**2))

rmse_Exp

##Quadratic model
Quad=smf.ols("Sales~t+t_squared",data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_squared"]]))
rmse_Quad=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))

rmse_Quad
################### Additive seasonality ########################

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))

rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_add_sea))**2))
rmse_add_sea
##Multiplicative seasonability model
mul_sea=smf.ols("log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul_sea = pd.Series(mul_sea.predict(Test))
rmse_mul_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul_sea)))**2))

rmse_mul_sea

################### Additive seasonality with quadratic trend ########################

add_sea_quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
##Multiplicative seasonability linear model
mul_add_sea=smf.ols("log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul_add_sea)))**2))

rmse_mul_add_sea
###let us create a dataframe and add all these rmse_values
data={"Model":pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
data
#Multiplicative seasonability linear model has got lowest value and accuracy better
## Now let us test the model with full data
model_full=smf.ols("log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=plastic1).fit()

predict_data = pd.read_excel("C:/360DG/Assignments/Time-series/plastic_predict.xlsx")

pred_new = pd.Series(model_full.predict(predict_data))
pred_new=np.exp(pred_new)
pred_new
predict_data["forecasted_Sales"] = pd.Series(pred_new)
#You check predict_data dataframe
#The benefits/impact of the solution - in what way does the business (client) benefit from the solution provided
#most previous injection molding machine studies have focused on R and D, production processes,
#and maintenance, with little consideration of sales activity. With the development and transformation
#of Industry 4.0 and the impact of the global economy,Injection molding machine industry
#growth rate has gradually flattened or even declined, with company sales and profits falling below
#expectations. Therefore, this study  understand the impact of economic indicators on injection molding sales. 

