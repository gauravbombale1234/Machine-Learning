# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:34:55 2024

@author: Gaurav Bombale
"""
import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

Walmart=pd.read_csv('C:/16_Time Series/Walmart Footfalls Raw.csv')
#data partition
Train=Walmart.head(147)
Test=Walmart.tail(12)
#In orderto use this model , we need to first find out values
#p represents number of Autoregressive terms - lags of dependent of p,d and q
#q represents number of moving average terms- lagged forecast errors in prediction equation
#d represents number of non-seasonal difference.


tsa_plots.plot_acf(Walmart.Footfalls, lags=12) # q for MA 5

tsa_plots.plot_pacf(Walmart.Footfalls, lags=12) 

#ARIMA with AR=3, MA=5
model1=ARIMA(Train.Footfalls, order=(3,1,5))
res1=model1.fit()
print(res1.summary())

#Forecast for next 12 month
start_index=len(Train)
end_index=start_index + 11
forecast_test=res1.predict(start=start_index,end=end_index)

print(forecast_test)

#Evaluate forecasts 
rmse_test=sqrt(mean_squared_error(Test.Footfalls, forecast_test))
print('Test RMSE: %.3f'%rmse_test)

#Plot forecast against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test, color='red')
pyplot.show()

#AUTO-ARIMA- Automatically discover the optimal order
#pip install pmdarima --user
import pmdarima as pm

ar_model=pm.auto_arima(Train.Footfalls, start_p=0, start_q=0, max_p=12, max_q=12, m=1, d=None,seasonal=False, start_P=0, trace=True, error_action='warn',stepwise=True)
#No Seasonality, m=1 means frequency of series