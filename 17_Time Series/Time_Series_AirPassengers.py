# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:33:51 2024

@author: Gaurav Bombale
"""
import numpy as  np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
plt.style.use('dark_background')

#Load the dataset
df=pd.read_csv('C:/16_Time Series/AirPassengers.csv')
df.columns
df=df.rename({'#Passengers':'Passengers'},axis=1)

print(df.dtypes)
#Month is text and passengers in int
#Now let us convert into data and time 
df['Month']=pd.to_datetime(df['Month'])
print(df.dtypes)

df.set_index('Month',inplace=True)

plt.plot(df.Passengers)
#There is increasing trend and  it has got seasonality 

#Is the data stationary?
#Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_=adfuller(df)
print('pvalue = ',pvalue,' if  above 0.05, data is not stationary')
#Since data is not stationary , we may need SARIMA and not just ARIMA
#Now let us extract the year and month from the date and time column

df['year']=[d.year for d in df.index]
df['month']=[d.strftime('%b') for d in df.index]
years=df['year'].unique()

#Plot yearly and monthly vallues as boxplot
sns.boxplot(x='year',y='Passengers',data=df)
#No. of passengers are going up year by year
sns.boxplot(x='month',y='Passengers',data=df)
#Over all there is higher trend in july and August compare to other

#Extract and plot trend , seaonal and residual
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed=seasonal_decompose(df['Passengers'],model='additive')

#Additive Time Series
#Value=Base level + Trend + Seasonality + Error
#Multiplicative Time Series
#Value=Base level * Trend * Seasonality * Error

trend=decomposed.trend
seasonal=decomposed.seasonal  #Cyclic behavior may not be seasonal
residual=decomposed.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passengers'],label='Original',color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend,label='Trend',color='yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal',color='yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual,label='Residual',color='yellow')
plt.legend(loc='upper left')

'''
Trend is going up from 1950s to 60s
It is highly seasonal showing peaks at particular interval
This helps to select specific prediction model
'''

#AUTOCORRELATION
#values are not corelated with x-axis but with its lag
#meaning yesturday's value is depend on day before yesturday's so on so forth 
#Autocorrelation is simply the correlation of a series with it's own lag
#Plot lag on x axis and correlation on y axis 
#Any correlation above confidence lines are statistically significant


from statsmodels.tsa.stattools import acf

acf_144=acf(df.Passengers,nlags=144)
plt.plot(acf_144)
#Autocorrelation above zero means positive correlation ans below as negative
#Obtain the same but with single line and more info...

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)
#any lag before 40 has positive correlation
#Horizontal bands indicate 95% and 99% (dashed) confidnece bands