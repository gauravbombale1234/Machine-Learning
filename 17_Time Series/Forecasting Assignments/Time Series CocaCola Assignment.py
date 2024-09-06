# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:54:41 2024

@author: Gaurav Bombale
"""
"""
2.	The dataset consists of quarterly sales data of Coca-Cola 
from 1986 to 1996. Predict sales for the next two years by using time 
series forecasting and prepare a document for each model explaining 
how many dummy variables you have created and also include the RMSE value
 for each model.
 
1.	Business Problem
1.1.	What is the business objective?
In this highly competitive business environment, forecasting becomes one of the
hot topics. Every business organization uses forecasts for decision marking.
Forecasting can help companies to determine the market strategy. It also helps in
production planning and resources allocation. A good forecast can help the
management team to make the best decision. Nowadays, it is important to develop
a collaborative partnership within the supply chain
1.2.	Are there any constraints?
Seasonality is one of the factors in the model. In different season, consumers tend
to have different buying habit. Nevertheless, the consumers’ buying pattern also
influence by the weather.
Moreover, the data of competitors’ actions are not available. The sales uplift may be
slightly lower when the same type of products of cocacola and its competitors are on
promotion at the same time,
"""
import pandas as pd
cocacola=pd.read_excel("CocaCola_Sales_RawData.xlsx")

cocacola.dtypes

p = cocacola["Quarter"][0]
p[0:2]
cocacola['Quarters']= 0    

for i in range(42):
    p = cocacola["Quarter"][i]
    cocacola["Quarters"][i]= p[0:2]
cocacola.drop(['Quarter'],axis=1,inplace=True)
cocacola=cocacola[['Quarters','Sales']]       
#########
    #for all these Quarters create dummy variables
quarter_dummies=pd.DataFrame(pd.get_dummies(cocacola['Quarters']))

# now let us concatenate these dummy values to dataframe
cocacola=pd.concat([cocacola,quarter_dummies],axis=1)

# you can check the dataframe cocacola

# similarly we need to create column t
import numpy as np
cocacola['t']=np.arange(1,43)
cocacola['t_squared']=cocacola['t']*cocacola['t']
cocacola['log_Sales']=np.log(cocacola['Sales'])
cocacola.columns
#Now let us check the visuals of the Sales
cocacola.Sales.plot()
#You will get increasing trend with linear increasing
# we have to forecast Sales in next 2 years,hence horizon=8,even 
#season=8,so validating data will be 8 and training will 42-8=34
Train=cocacola.head(34)
Test=cocacola.tail(8)
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
#Q("1995")+Q("1996")+Q("1997")+Q("1998")+Q("1999")+Q("2000")+Q("2001")+Q("2002")
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_add_sea))**2))
rmse_add_sea
##Multiplicative seasonability model
mul_sea=smf.ols("log_Sales~Q1+Q2+Q3+Q4",data=Train).fit()
pred_mul_sea = pd.Series(mul_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_mul_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul_sea)))**2))

rmse_mul_sea

################### Additive seasonality with quadratic trend ########################
add_sea_quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4', data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
##Multiplicative seasonability linear model
mul_add_sea=smf.ols("log_Sales~t+Q1+Q2+Q3+Q4",data=Train).fit()
pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul_add_sea)))**2))
rmse_mul_add_sea

###let us create a dataframe and add all these rmse_values
data={"Model":pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
data
#rmse_quad has lowest value means better accuracy,Additive seasonality with quadratic trend has lowest value
## Now let us test the model with full data

model_full = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4', data=cocacola).fit()

cocacola_predict=pd.read_excel("C:/360DG/assignments/Time-series/cocacola_predict.xlsx")
pred_new = pd.Series(add_sea_quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))

cocacola_predict["t_squared"]=cocacola_predict["t"]*cocacola_predict["t"]
cocacola_predict.to_csv("cocacola_predict.csv",encoding="utf-8")
import os
os.getcwd()
predict_data = pd.read_csv("C:/360DG/assignments/Time-series/cocacola_predict.csv")
pred_new = pd.Series(model_full.predict(predict_data[['Q1','Q2','Q3','Q4','t','t_squared']]))

predict_data["forecasted_Sales"] = pd.Series(pred_new)

#Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided
#The quality of the input data is important to the forecasts. Manufactures are able to develop a
#high accuracy forecast when their input data comes from its customers, i.e. the
#wholesalers and retailers.The order history record may
#mislead the forecasts due to the poor forecasts from wholesalers and retailers. 