# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:48:53 2024

@author: Admin
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Now load the dataset
cocacola=pd.read_excel('C:/16_Time Series/Cocacola_Sales_Rawdata.xlsx')
#Let us plot the dataset and its nature
cocacola.Sales.plot()
#Splittingthe data into train and test set data
#since we are working on quaterly datasets and in year there are 4 quarters
#Test data = 4 quarters
#train data=38
Train=cocacola.head(38)
Test=cocacola.tail(4)
#Here we are considering performance parameters as mean absolute
#rather than mean square error
#custom function is written to calculate MPSE
def MAPE(pred, org):
    temp=np.abs((pred-org)/org)*100
    return np.mean(temp)

#EDA which comprises identification of level , trends, and seasonality
#In order to separate trend and seasonality moving average can 
my_pred=cocacola['Sales'].rolling(4).mean()
my_pred.tail(4)
#now let us calculate mean absolute percentage of this 
#values
MAPE(my_pred.tail(4),Test.Sales)
#Moving average is predicting complete values, out of which last
#Are considered as predicted values and last four values of Test
#Basic purpose of moving average is deseasonalizing

cocacola.Sales.plot(label='org')
#This is original plot
#Now let us separate out Trend and Seasonality 
for i in range(2,9,2):
    #it will take window size 2,4,6,8
    cocacola['Sales'].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=3)
#you can see i=4 and 8 are deseasonable plots

#Time series decomposition is the another technique of separating
#seasonality 
decompose_ts_add=seasonal_decompose(cocacola.Sales, model='additive', period=4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

#similar plot can be decomposed using multiplication
decompose_ts_mul=seasonal_decompose(cocacola.Sales, model='multiplicative',period=4)
print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()

#you can observed the difference between these plots
#Now let us plot ACF plot to check the auto correlation
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags=4)
#we can observed the output in which r1,r2,r3 and r4 has higher
#This is all about EDA
#Let us apply data to data driven models
#Simple Exponential Method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model=SimpleExpSmoothing(Train['Sales']).fit()
pred_ses=ses_model.predict(start=Test.index[0],end=Test.index[-1])
#Now calculate MAPE
MAPE(pred_ses, Test.Sales)
#We are getting 8.30789775441469
#Holts exponential smoothing #here only trend is captured
hw_model=Holt(Train['Sales']).fit()
pred_hw=hw_model.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hw,Test.Sales)
#9.809783428618136

#Holts winter exponential snoothing with additive seasonality
hwe_model_add_add=ExponentialSmoothing(Train['Sales'],seasonal='add',trend='add',seasonal_periods=4).fit()
pred_hwe_model_add_add=hwe_model_add_add.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hwe_model_add_add,Test.Sales)
# 1.5022364186549284
#Holts winter seasonal exponential smoothing with multiplicative seasinalty
hwe_model_mul_add=ExponentialSmoothing(Train['Sales'],seasonal='mul',trend='add',seasonal_periods=4).fit()
pred_hwe_model_mul_add=hwe_model_mul_add.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hwe_model_mul_add,Test.Sales)
