# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:22:23 2024

@author: Gaurav Bombale
"""

"""
4.With the growing consumption of avocados in the USA, a freelance company 
would like to do some analysis on the patterns of consumption in different 
cities and would like to come up with a prediction model for the price of 
avocados. For this to be implemented, build a prediction model using 
multilinear regression and provide your insights on it.

1.1.	What is the business objective?
The data represents weekly 2018 retail scan data for National retail 
volume (units) and price. Retail scan data comes directly from 
retailers’ cash registers based on actual retail sales of Hass avocados.
Starting in 2013, the data reflects an expanded, 
multi-outlet retail data set. Multi-outlet reporting includes an 
aggregation of the following channels: grocery, mass, club, drug, dollar and military. The Average Price (of avocados) in the data reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. The Product Lookup codes (PLU’s) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this data.


1.2.	Are there any constraints?
"""
# Dataset
#AveragePrice: The average price of a single avocado
#Total Volume: Total number of avocados sold
#Total Bags
#Small Bags
#Large Bags
#XLarge Bags
#type: conventional or organic
#year: The year
#region: The city or region of the observation

import pandas as pd
import numpy as np
import seaborn as sns
ava=pd.read_csv("Avacado_Price.csv")
ava_new=ava.iloc[:,0:11]
# Exploratory data analysis
#1.Measure the central tendency
#2.Measure the dispersion
#3.Third moment business decision
#4.Fourth moment business decision
#5.probability distribution
#6.Graphical represenation(Histogram,Boxplot)

ava_new.describe()
#Following columns have been dropped
#Total Volume:sum of Hass Avocado with PSU labels 
#Total Bags:sum of various bag size
#date column has been dropped as there too many levels
#region column has been dropped as there are too many levels
ava_new=ava_new.rename(columns={'XLarge Bags':'XLarge_Bags'})
ava_new.isna().sum()
#There are no null values
import matplotlib.pyplot as plt
plt.bar(height=ava_new.AveragePrice,x=np.arange(1,18250,1))
sns.distplot(ava_new.AveragePrice)
#Data is normal slight right skewed
plt.boxplot(ava_new.AveragePrice)
# There are several outliers
plt.bar(height=ava_new.Total_Volume,x=np.arange(1,18250,1))
sns.distplot(ava_new.Total_Volume)
#Data is normal but right skewed
plt.boxplot(ava_new.Total_Volume)
#There are several outliers
#let us check tot_ava1
plt.bar(height=ava_new.tot_ava1,x=np.arange(1,18250,1))
sns.distplot(ava_new.tot_ava1)
#Data is normal but slight right skewed
plt.boxplot(ava_new.tot_ava1)
#There are several outliers
# let us check tot_ava2
plt.bar(height=ava_new.tot_ava2,x=np.arange(1,18250,1))
sns.distplot(ava_new.tot_ava2)
#Data is normal but slight right skewed
plt.boxplot(ava_new.tot_ava2)
#There are several outliers
# let us check tot_ava3
plt.bar(height=ava_new.tot_ava3,x=np.arange(1,18250,1))
sns.distplot(ava_new.tot_ava3)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.tot_ava3)
#There are several outliers
#let us check Total_Bags
plt.bar(height=ava_new.Total_Bags,x=np.arange(1,18250,1))
sns.distplot(ava_new.Total_Bags)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.Total_Bags)
#There are several outliers

#let us check Small_Bags
plt.bar(height=ava_new.Small_Bags,x=np.arange(1,18250,1))
sns.distplot(ava_new.Small_Bags)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.Small_Bags)
#There are several outliers
#let us check Large_Bags
plt.bar(height=ava_new.Large_Bags,x=np.arange(1,18250,1))
sns.distplot(ava_new.Large_Bags)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.Large_Bags)
#There are several outliers

######################################################
###Data preprocessing

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
ava_new["year"]=lb.fit_transform(ava["year"])
ava_new["type"]=lb.fit_transform(ava["type"])
ava_new.dtypes
from feature_engine.outliers import Winsorizer
import seaborn as sns
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['AveragePrice'])
ava_t=winsor.fit_transform(ava_new[['AveragePrice']])
sns.boxplot(ava_t.AveragePrice)
ava_new['AveragePrice']=ava_t['AveragePrice']
plt.boxplot(ava_new.AveragePrice)
# let us check Total_Volume
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Total_Volume'])
ava_t=winsor.fit_transform(ava_new[['Total_Volume']])
sns.boxplot(ava_t.Total_Volume)
ava_new['Total_Volume']=ava_t['Total_Volume']
plt.boxplot(ava_new.Total_Volume)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava1'])
ava_t=winsor.fit_transform(ava_new[['tot_ava1']])
sns.boxplot(ava_t.tot_ava1)
ava_new['tot_ava1']=ava_t['tot_ava1']
plt.boxplot(ava_new.tot_ava1)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava2'])
ava_t=winsor.fit_transform(ava_new[['tot_ava2']])
sns.boxplot(ava_t.tot_ava2)
ava_new['tot_ava2']=ava_t['tot_ava2']
plt.boxplot(ava_new.tot_ava2)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava3'])
ava_t=winsor.fit_transform(ava_new[['tot_ava3']])
sns.boxplot(ava_t.tot_ava3)
ava_new['tot_ava3']=ava_t['tot_ava3']
plt.boxplot(ava_new.tot_ava3)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Total_Bags'])
ava_t=winsor.fit_transform(ava_new[['Total_Bags']])
sns.boxplot(ava_t.Total_Bags)
ava_new['Total_Bags']=ava_t['Total_Bags']
plt.boxplot(ava_new.Total_Bags)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Small_Bags'])
ava_t=winsor.fit_transform(ava_new[['Small_Bags']])
sns.boxplot(ava_t.Small_Bags)
ava_new['Small_Bags']=ava_t['Small_Bags']
plt.boxplot(ava_new.Small_Bags)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Large_Bags'])
ava_t=winsor.fit_transform(ava_new[['Large_Bags']])
sns.boxplot(ava_t.Large_Bags)
ava_new['Large_Bags']=ava_t['Large_Bags']
plt.boxplot(ava_new.Large_Bags)




#Graphical represenation,Bivariant analysis

#Now let us check colinearity between Y and X1,X2,....plot joint plot,joint plot is to show scatter plot as well 
# histogram
ava_new.dtypes
import seaborn as sns
sns.jointplot(x=ava_new['Total_Volume'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['tot_ava1'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['tot_ava2'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['tot_ava3'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['Total_Bags'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['Small_Bags'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['Large_Bags'],y=ava_new['AveragePrice'])
sns.jointplot(x=ava_new['XLarge_Bags'],y=ava_new['AveragePrice'])
#################################################
import numpy as np

corr_df = ava_new.corr(method='pearson')
plt.figure(figsize=(12,6),dpi=100)
sns.heatmap(corr_df,cmap='coolwarm',annot=True)
#From heatmap it is clear that only AveragePrice and type are good correlated
#Rest all are poorly correlated
#Let us check colinearity between X and X
# Total_Volume is correlated with all other varibles
#Except X_large_bags, Type and year,  

# tot_ava1 is correlated with all other varibles
#Except X_large_bags, Type and year, 

# tot_ava2 is correlated with all other varibles
#Except X_large_bags, Type and year, 

# tot_ava3 is correlated with all other varibles
#Except X_large_bags, Type and year, 
# Total_Bags is correlated with all other varibles
#Except X_large_bags, Type and year, 

# Small_Bags is correlated with all other varibles
#Except X_large_bags, Type and year, 
# Large_Bags is correlated with all other varibles
#Except X_large_bags, Type and year, 
##########
#only type and year are not correlated with other variables.

ava_new.dtypes
##QQ plot
from scipy import stats
import pylab
stats.probplot(ava_new['AveragePrice'],dist="norm",plot=pylab)
#Data is normal
stats.probplot(ava_new['Total_Volume'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['tot_ava1'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['tot_ava2'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['tot_ava3'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['Total_Bags'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['Small_Bags'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['Large_Bags'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(ava_new['XLarge_Bags'],dist="norm",plot=pylab)
#Data is normal

plt.show()
# Average Price data is normally distributed
# There are 28 scatter plots need to be plotted,one by one is difficult
#to plot,so we can use pair plots
import seaborn as sns
sns.pairplot(ava_new.iloc[:,:])
# you can check the collinearity problem between the input variables
# you can check plot between Total_ava1 and Total_ava2,they are strongly corelated
# same way you can check WT and VOL,it is also strongly correlated

# now let us check r value between variables
ava_new.dtypes
ava_new.corr()
#Except type and year,all other variables are poorly correlated with AveragePrice
#Except X_large_bags,type and year,all other variables are correlated with each other 
import statsmodels.formula.api as smf
rsq_tot_vol=smf.ols('Total_Volume~tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_tot_vol=1/(1-rsq_tot_vol)

rsq_tot_ava1=smf.ols('tot_ava1~Total_Volume+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_tot_ava1=1/(1-rsq_tot_ava1)

rsq_tot_ava2=smf.ols('tot_ava2~Total_Volume+tot_ava1+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_tot_ava2=1/(1-rsq_tot_ava2)

rsq_tot_ava3=smf.ols('tot_ava3~Total_Volume+tot_ava1+tot_ava2+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_tot_ava3=1/(1-rsq_tot_ava3)

rsq_tot_bags=smf.ols('Total_Bags~tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_tot_bags=1/(1-rsq_tot_bags)

rsq_Small_Bags=smf.ols('Small_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_Small_Bags=1/(1-rsq_Small_Bags)

rsq_Large_Bags=smf.ols('Large_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+XLarge_Bags+type+year',data=ava_new).fit().rsquared
vif_Large_Bags=1/(1-rsq_Large_Bags)

rsq_XLarge_Bags=smf.ols('XLarge_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+Large_Bags+type+year',data=ava_new).fit().rsquared
vif_XLarge_Bags=1/(1-rsq_XLarge_Bags)

rsq_type=smf.ols('type~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+Large_Bags+XLarge_Bags+year',data=ava_new).fit().rsquared
vif_type=1/(1-rsq_type)

rsq_year=smf.ols('Large_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+XLarge_Bags+type',data=ava_new).fit().rsquared
vif_year=1/(1-rsq_year)




d1={'Variables':['Total_Volume','tot_ava1','tot_ava2','tot_ava3','Total_Bags','Small_Bags','Large_Bags','XLarge_Bags','type','year'],
    'VIF':[vif_tot_vol,vif_tot_ava1,vif_tot_ava2,vif_tot_ava3,vif_tot_bags,vif_Small_Bags,vif_Large_Bags,vif_XLarge_Bags,vif_type,vif_year]}

vif_frame=pd.DataFrame(d1)
vif_frame
#Total_Volume,tot_av2,Total_Bags and Small_Bags have vif>10 hence dropping these columns
import statsmodels.formula.api as smf
ml=smf.ols('AveragePrice~tot_ava1+tot_ava3+Large_Bags+XLarge_Bags+type+year',data=ava_new).fit()
ml.summary()
# R-Square is 0.438<0.85 and p-values=0
#Y='AveargePrice'
#
#X1=np.log(ava_new.tot_ava1)

#X2=np.log(ava_new.tot_ava3)
#X3=np.log(ava_new.Large_Bags)
#X4=np.log(ava_new.XLarge_Bags)

#ava_new.dropna(inplace=True)

#ml2=smf.ols('AveragePrice~X1+X2+X3+X4+type+year',data=ava_new).fit()
#ml2.summary()



# prediction
pred=ml.predict(ava_new)
import statsmodels.api as sm
##QQ plot
res=ml.resid
sm.qqplot(res)
plt.show()
# This QQ plot is on residual which is obtained on training data
#eerors are obtained on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

#let us plot the residual plot ,which takes the residuals values 
#and the data
sns.residplot(x=pred,y=ava_new.AveragePrice,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
# residual plots are used to check whether the errors are independent or not

# let us plot the influence plot
sm.graphics.influence_plot(ml)

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
ava_train,ava_test=train_test_split(ava_new,test_size=0.2)
#preparing the model on train data 
model_train=smf.ols('AveragePrice~tot_ava1+tot_ava3+Large_Bags+XLarge_Bags+type+year',data=ava_train).fit()
model_train.summary()
test_pred=model_train.predict(ava_test)
##test_errors
test_error=test_pred-ava_test.AveragePrice
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse

