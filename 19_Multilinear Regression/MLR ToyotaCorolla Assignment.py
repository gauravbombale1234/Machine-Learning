# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:22:22 2024

@author: Gaurav Bombale
"""
"""
3.	An online car sales platform would like to improve its customer base 
and their experience by providing them an easy way to buy and sell cars. 
For this, they would like an automated model which can predict the price
of the car once the user inputs the required factors. Help the business 
achieve their objective by applying multilinear regression on the given 
dataset. Please use the below columns for the analysis purpose: price, 
age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, and Weight.
"""

# multiple correlation regression analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
cars=pd.read_csv("ToyotaCorolla.csv",encoding='ISO-8859-1')
# There is model column which will have to drop
cars.drop(['Model','Id'],axis=1,inplace=True)
cars_new=cars.iloc[:,[0,1,4,6,10,11,13,14,15]]
# Exploratory data analysis
cars_new.describe()
#data is right skewed
import matplotlib.pyplot as plt
import seaborn as sns
plt.bar(height=cars_new.Price,x=np.arange(1,1437,1))
plt.hist(cars_new.Price)
#Price is right skewed
plt.boxplot(cars_new.Price)
##There are several outliers
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Price'])
cars_t=winsor.fit_transform(cars_new[['Price']])
sns.boxplot(cars_t.Price)
cars_new['Price']=cars_t['Price']
plt.boxplot(cars_new.Price)
# let us rename column age_08_04 as age
cars_new=cars_new.rename(columns={'Age_08_04':'Age'})

plt.bar(height=cars_new.Age,x=np.arange(1,1437,1))
plt.hist(cars_new.Age)
#Price is right skewed
plt.boxplot(cars_new.Age)
#There are several outliers
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Age'])
cars_t=winsor.fit_transform(cars_new[['Age']])
sns.boxplot(cars_t.Age)
cars_new['Age']=cars_t['Age']
plt.boxplot(cars_new.Age)
### let us check KM
plt.bar(height=cars_new.KM,x=np.arange(1,1437,1))
plt.hist(cars_new.KM)
#KM is right skewed
plt.boxplot(cars_new.KM)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['KM'])
cars_t=winsor.fit_transform(cars_new[['KM']])
sns.boxplot(cars_t.KM)
cars_new['KM']=cars_t['KM']
plt.boxplot(cars_new.KM)
#Now let us check HP
plt.bar(height=cars_new.HP,x=np.arange(1,1437,1))
plt.hist(cars_new.HP)
#KM is right skewed
plt.boxplot(cars_new.HP)
# there is one outlier
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['HP'])
cars_t=winsor.fit_transform(cars_new[['HP']])
sns.boxplot(cars_t.HP)
cars_new['HP']=cars_t['HP']
plt.boxplot(cars_new.HP)

##Now let us check cc
plt.bar(height=cars_new.cc,x=np.arange(1,1437,1))
plt.hist(cars_new.cc)
#
plt.boxplot(cars_new.cc)
#There are outliers
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['cc'])
cars_t=winsor.fit_transform(cars_new[['cc']])
sns.boxplot(cars_t.cc)
cars_new['cc']=cars_t['cc']
plt.boxplot(cars_new.cc)
# Now let us check the Doors column
plt.bar(height=cars_new.Doors,x=np.arange(1,1437,1))
plt.hist(cars_new.Doors)
#
plt.boxplot(cars_new.Doors)
#There are no outliers

# let us check Gears
plt.bar(height=cars_new.Gears,x=np.arange(1,1437,1))
plt.hist(cars_new.Gears)
#
plt.boxplot(cars_new.Gears)
#There are outliers

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Gears'])
cars_t=winsor.fit_transform(cars_new[['Gears']])
sns.boxplot(cars_t.Gears)
cars_new['Gears']=cars_t['Gears']
plt.boxplot(cars_new.Gears)
# Now let us check the Quarterly_Tax
plt.bar(height=cars_new.Quarterly_Tax,x=np.arange(1,1437,1))
plt.hist(cars_new.Quarterly_Tax)
#
plt.boxplot(cars_new.Quarterly_Tax)
# there are outliers
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Quarterly_Tax'])
cars_t=winsor.fit_transform(cars_new[['Quarterly_Tax']])
sns.boxplot(cars_t.Quarterly_Tax)
cars_new['Quarterly_Tax']=cars_t['Quarterly_Tax']
plt.boxplot(cars_new.Quarterly_Tax)

# Now let us check Weight
plt.bar(height=cars_new.Weight,x=np.arange(1,1437,1))
plt.hist(cars_new.Weight)
#
plt.boxplot(cars_new.Weight)
# there are outliers
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Weight'])
cars_t=winsor.fit_transform(cars_new[['Weight']])
sns.boxplot(cars_t.Weight)
cars_new['Weight']=cars_t['Weight']
plt.boxplot(cars_new.Weight)
#1.Measure the central tendency
#2.Measure the dispersion
#3.Third moment business decision
#4.Fourth moment business decision
#5.probability distribution
#6.Graphical represenation(Histogram,Boxplot)
cars.describe()
#Graphical represenation

##
# similar oerations are expected for other three columns

#Now let us plot joint plot,joint plot is to show scatter plot as well 
# histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['Price'])

# now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times the each value occured
#92 HP value occured 7 times

##QQ plot
from scipy import stats
import pylab
stats.probplot(cars.Price,dist="norm",plot=pylab)
plt.show()
#  Price data is moderately normally distributed
# There are 36 scatter plots need to be plotted,one by one is difficult
#to plot,so we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
# you can check the collinearity problem between the input variables
# you can check plot between SP and HP,they are strongly corelated
# same way you can check WT and VOL,it is also strongly correlated

# now let us check r value between variables
cars.corr()
# you can check SP and HP,r value is 0.97 and same way
# you can check WT and VOL ,it has got 0.999 which is greater

# Now although we observed strogly correlated pairs,still we will go for 
# linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=cars_new).fit()
ml1.summary()
#R square value observed is 0.867<0.85 
#p-values of are up to the mark 

import statsmodels.api as sm
# prediction
pred=ml1.predict(cars_new)

##QQ plot
res=ml1.resid
sm.qqplot(res)
plt.show()
# This QQ plot is on residual which is obtained on training data
#eerors are obtained on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

#let us plot the residual plot ,which takes the residuals values 
#and the data
sns.residplot(x=pred,y=cars.Price,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
# residual plots are used to check whether the errors are independent or not

# let us plot the influence plot
sm.graphics.influence_plot(final_ml)
# we have taken cars instead car_new data ,hence 76 is reflected
# again in influencial data

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test=train_test_split(cars_new,test_size=0.2)
#preparing the model on train data 
model_train=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=cars_train).fit()
model_train.summary()
#R-Squared is 0.866 and p values are 0
test_pred=model_train.predict(cars_test)
##test_errors
test_error=test_pred-cars_test.Price
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse


