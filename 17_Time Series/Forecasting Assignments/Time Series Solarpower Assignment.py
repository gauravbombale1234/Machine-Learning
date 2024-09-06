# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:54:42 2024

@author: Gaurav Bombale
"""
"""
4. Solar power consumption has been recorded by city councils at regular 
intervals. The reason behind doing so is to understand how businesses are
using solar power so that they can cut down on nonrenewable sources of 
energy and shift towards renewable energy. Based on the data, build a 
forecasting model and provide insights on it. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


solar = pd.read_csv("Solarpower.csv")

solar.cum_power.plot() # time series plot 
#linearly increasing slight trend and season
# Splitting the data into Train and Test data
# Recent 4 time period values are Test data,2558-7=2551
Train = solar.head(2551)
Test = solar.tail(7)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Moving Average for the time series
mv_pred = solar["cum_power"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), Test.cum_power)


# Plot with Moving Averages
solar.cum_power.plot(label = "org")
for i in range(2, 9, 2):
    solar["cum_power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(solar.cum_power, model = "additive", period = 7)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(solar.cum_power, model = "multiplicative", period = 7)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(solar.cum_power, lags = 4)
tsa_plots.plot_pacf(solar.cum_power, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["cum_power"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.cum_power) 

# Holt method 
hw_model = Holt(Train["cum_power"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.cum_power) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.cum_power) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.cum_power) 

#Holts method has lowest MPSE value
# Final Model on 100% Data


hw_model_full = Holt(solar["cum_power"]).fit()
pred_hw_new = hw_model.predict(start = solar.index[0], end = solar.index[-1])
pred_hw_new


# Load the new data which includes the entry for future 4 values
new_data = pd.read_csv("solarpower_pred_new.csv")

newdata_pred = hw_model_full.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

##################################################