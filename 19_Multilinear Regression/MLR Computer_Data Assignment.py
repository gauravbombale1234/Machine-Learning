# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:22:20 2024

@author: Gaurav Bombale
"""

"""
2.	Perform multilinear regression with price as the output variable 
and document the different RMSE values.

# multiple correlation regression analysis
1.1.	What is the business objective?
       One reason both analyses mentioned is higher
       computer prices for consumers, which has been noted 
       was due to higher prices for certain components, 
       namely memory chips and other perpherals ,The main objective
       is to establish relation among these with price of computer.
1.2.	Are there any constraints?
     Is the price competitive?  
       What type of discount (e.g., trade, cash, quantity) and allowances (e.g., advertising, trade-offs) should your company offer its foreign customers?  
      Should prices differ by market segment?  
      What should your company do about product-line pricing?  
"""
#Dataset
#price          int64
#speed          int64
#hd             int64
#ram            int64
#screen         int64
#cd            object
#multi         object
#premium       object
#ads            int64
#trend          int64

import pandas as pd
import numpy as np
comp=pd.read_csv("Computer_Data.csv")
comp.columns
comp.dtypes
#There are three columns which are object type need to convert
#dummy variables
comp.drop(['Unnamed: 0'],axis=1,inplace=True)
comp_new=pd.get_dummies(comp,drop_first=True)
# check comp_new dataframe
comp_new=comp_new.drop_duplicates()
comp_new
#There are no duplicate intries

# Exploratory data analysis
#1.Measure the central tendency
#2.Measure the dispersion
#3.Third moment business decision
#4.Fourth moment business decision
#5.probability distribution
#6.Graphical represenation(Histogram,Boxplot)
comp_new.price.describe()
#average price is 221.83 and min 947 and max is 5399
comp_new.speed.describe()
#average speed is 52.129549 and min 25.00 and max is 100
comp_new.hd.describe()
#average speed is 417.76 and min 80 and max is 2100


#Graphical represenation
import matplotlib.pyplot as plt
import seaborn as sns
plt.bar(height=comp_new.price,x=np.arange(1,6184,1))
sns.distplot(comp_new.price)
#Price  normal but is right skewed
plt.boxplot(comp_new.price)
#There are several outliers
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['price'])
comp_t=winsor.fit_transform(comp_new[['price']])
sns.boxplot(comp_t.price)
comp_new['price']=comp_t['price']
plt.boxplot(comp_new.price)
##let us check the speed column
plt.bar(height=comp_new.speed,x=np.arange(1,6184,1))
sns.distplot(comp_new.price)
# speed column data is almost normal
plt.boxplot(comp_new.speed)
##There are no outliers in speed columns,slight right skewed
# similar oerations are expected for other three columns
plt.bar(height=comp_new.hd,x=np.arange(1,6184,1))
sns.distplot(comp_new.hd)
# hd column data is right skewed
plt.boxplot(comp_new.hd)
# there are several outliers in hd coulmn,need to treat
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['hd'])
comp_hd=winsor.fit_transform(comp_new[['hd']])
sns.boxplot(comp_hd.hd)
comp_new['hd']=comp_hd['hd']
plt.boxplot(comp_new.hd)
##let us check ram
plt.bar(height=comp_new.ram,x=np.arange(1,6184,1))
sns.distplot(comp_new.ram)
# ram  column data is normal but right skewed
plt.boxplot(comp_new.ram)
# there are several outliers in hd coulmn,need to treat
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ram'])
comp_ram=winsor.fit_transform(comp_new[['ram']])
sns.boxplot(comp_ram.ram)
comp_new['ram']=comp_ram['ram']

plt.boxplot(comp_new.ram)
###let us check the screen column

plt.boxplot(comp_new.screen)
##let us check screen
plt.bar(height=comp_new.screen,x=np.arange(1,6184,1))
plt.hist(comp_new.screen)
plt.boxplot(comp_new.screen)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['screen'])
comp_screen=winsor.fit_transform(comp_new[['screen']])
sns.boxplot(comp_screen.screen)
comp_new['screen']=comp_screen['screen']
plt.boxplot(comp_new.screen)

########let us check cd_yes
plt.bar(height=comp_new.cd_yes,x=np.arange(1,6184,1))
plt.hist(comp_new.cd_yes)
plt.boxplot(comp_new.cd_yes)

##let us check ads
plt.bar(height=comp_new.ads,x=np.arange(1,6184,1))
plt.hist(comp_new.ads)
plt.boxplot(comp_new.ads)
# ads data is outliers free

##let us check trend
plt.bar(height=comp_new.trend,x=np.arange(1,6184,1))
plt.hist(comp_new.trend)
plt.boxplot(comp_new.trend)
# trend data is outliers free

##let us check multi_yes
plt.bar(height=comp_new.multi_yes,x=np.arange(1,6184,1))
plt.hist(comp_new.multi_yes)
plt.boxplot(comp_new.multi_yes)
# trend data is outliers free
##let us check premium_yes
plt.bar(height=comp_new.premium_yes,x=np.arange(1,6184,1))
plt.hist(comp_new.premium_yes)
plt.boxplot(comp_new.premium_yes)
# premium_yes data is outliers free
####data preprocessing is done


#Now let us plot joint plot,joint plot is to show scatter plot as well 
# histogram
import seaborn as sns
sns.jointplot(x=comp_new['price'],y=comp_new['speed'])

# now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(comp_new['price'])
#count plot shows how many times the each value occured

##QQ plot
from scipy import stats
import pylab
stats.probplot(comp_new.price,dist="norm",plot=pylab)
plt.show()
# price data is normally distributed
# There are 45 scatter plots need to be plotted,one by one is difficult
#to plot,so we can use pair plots
import seaborn as sns
sns.pairplot(comp_new.iloc[:,:])
# you can check the collinearity problem between the input variables
#

# now let us check r value between variables
comp_new.corr()

# Now although we observed strongly correlated pairs,still we will go for 
# linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_new).fit()
ml1.summary()
#R-squared: 0.782<0.85
#p-values 0 for all which is desired 
# it means it is less than 0.05,

final_ml=smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_new).fit()
final_ml.summary()
#R square is 0.770 and p values 0.00,0.012 <0.05

# prediction
pred=final_ml.predict(comp_new)
import statsmodels.api as sm
##QQ plot
res=final_ml.resid
sm.qqplot(res)
plt.show()
# This QQ plot is on residual which is obtained on training data
#errors are obtained on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

#let us plot the residual plot ,which takes the residuals values 
#and the data
sns.residplot(x=pred,y=comp_new.price,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
# residual plots are used to check whether the errors are independent or not

# let us plot the influence plot
sm.graphics.influence_plot(final_ml)

# again in influencial data

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
comp_train,comp_test=train_test_split(comp_new,test_size=0.2)
#preparing the model on train data 
model_train=smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_train).fit()
model_train.summary()
test_pred=model_train.predict(comp_test)
##test_errors
test_error=test_pred-comp_test.price
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse
# train_data prediction
train_pred = model_train.predict(comp_train)

# train residual values 
train_resid  = train_pred - comp_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

