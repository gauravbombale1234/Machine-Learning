# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:22:20 2024

@author: Gaurav Bombale
"""
'''
1.	An analytics company has been tasked with the
 crucial job of finding out what factors affect 
 a startup company and if it will be profitable 
 or not. For this, they have collected some 
 historical data and would like to apply multilinear
 regression to derive brief insights into their data.
 Predict profit, given different attributes for various startup companies.

1.	Business Problem
1.1.	What is the business objective?
Profitability in business is a matter of survival: 
    If your business doesn't stay profitable, 
    you don't stay in business. The simple 
    definition of profitability is that your revenue
    is more than your expenses.
    The objective of this is to find impact of research, administration ,Marketing Spend on profit
1.2.	Are there any constraints?
      The corporate world is quite fierce. 
      There is always a competition going on between
      the giants.
      There is a huge pool of aspiring individuals
      available. Selecting a suitable candidate 
      that fits the job well enough is a peculiarly
      tricky task.
      Customer is the king. And that’s absolutely 
      right. Winning a customer’s trust is one of 
      the most important challenges that businesses 
      in general – and startups in particular
'''
# Database Description
#Research          float64
#Administration    float64
#Marketing         float64
#Profit            float64
import pandas as pd
import numpy as np
import seaborn as sns
# loading the data
start_up = pd.read_csv("50_Startups.csv")
start_up=start_up.rename(columns={'R&D Spend':'Research','Marketing Spend':'Marketing'})
# Exploratory data analysis:
# 1. Measures of central tendency
start_up.dtypes
start_up.describe()
#average cost for resaerch is 73721 and 
#min=0 and max is 165349
#avearge cost of administration is 121344
#min=51283 and max=182645
##Data is right skewed
#average cost for marketing is 211025 and 
#min=0 and max is 471784
#average profit is 112012 and 
#min=14681 and max is 192261

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
# R and D spend
plt.bar(height = start_up['Research'], x = np.arange(1, 51, 1))
sns.distplot(start_up['Research']) #histogram
# Data is almost normallly distributed
plt.boxplot(start_up['Research']) #boxplot
# No ouliers
plt.bar(height = start_up['Administration'], x = np.arange(1, 51, 1))
sns.distplot(start_up['Administration']) #histogram
# Data is almost normal slight left skewed
plt.boxplot(start_up['Administration']) #boxplot
#No outliers
plt.bar(height = start_up['Marketing'], x = np.arange(1, 51, 1))
sns.distplot(start_up['Marketing']) #histogram
# Data is almost normal
plt.boxplot(start_up['Marketing']) #boxplo
# No outliers

plt.bar(height = start_up['Profit'], x = np.arange(1, 51, 1))
plt.hist(start_up['Profit']) #histogram
# Data is almost normal
plt.boxplot(start_up['Profit']) #boxplo
# There is one outlier
# 
# Jointplot
import seaborn as sns
sns.jointplot(x=start_up['Research'], y=start_up['Profit'])
#Resaerch and profit are linear
sns.jointplot(x=start_up['Administration'], y=start_up['Profit'])
# There is weak linearity of admin cost and profit
sns.jointplot(x=start_up['Marketing'], y=start_up['Profit'])
# Marketing spend and profit are almost linear
start_up.drop(['State'],axis=1,inplace=True)
# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(start_up['Research'])
sns.countplot(start_up['Administration'])
sns.countplot(start_up['Marketing'])
# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(start_up.Profit, dist = "norm", plot = pylab)
plt.show()
# Data is normallly distributed

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(start_up.iloc[:, :])
#Profit and Administration is nonliear 
#except marketing spend and R & D spend rest all are not linear
                          
# Correlation matrix 
start_up.corr()
# There is high colinearity between Profit n R &D spend
#similarly profit and marketing spend it is desired
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
first_model = smf.ols('Profit ~Research + Administration+ Marketing', data = start_up).fit() # regression model

# Summary
first_model.summary()
# R-squared: 0.951,p-values for administration is 0.602 more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(first_model)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

start_up_new = start_up.drop(start_up.index[[49]])

# Preparing model                  
first_model_new = smf.ols('Profit ~Research + Administration+ Marketing', data = start_up_new).fit()    

# Summary
first_model_new.summary()
# There is no change in p value of administration

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF should not be > 10 = colinearity
# calculating VIF's values of independent variables
rsq_research = smf.ols('Research ~ Administration +Marketing', data = start_up).fit().rsquared  
vif_research = 1/(1 - rsq_research) 

rsq_administration = smf.ols('Administration ~ Research +Marketing', data = start_up).fit().rsquared  
vif_administration = 1/(1 - rsq_administration)

rsq_marketing = smf.ols('Marketing ~Research +Administration', data = start_up).fit().rsquared  
vif_marketing = 1/(1 - rsq_marketing) 

# Storing vif values in a data frame
d1 = {'Variables':['Research', 'Administration', 'Marketing',], 'VIF':[vif_research, vif_administration, vif_marketing]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# All are having less than 10 VIF value
# let us go for transformation
new_admin=np.log(start_up['Administration'])
# Final model
trans_model = smf.ols('Profit ~ Research +new_admin + Marketing', data = start_up).fit()
trans_model.summary() 
##New_admin has p value=0.689 which is not improved hence Administration feature has to be droped
# Prediction
final_ml = smf.ols('Profit ~ Research + Marketing', data = start_up).fit()
final_ml.summary() 
pred = final_ml.predict(start_up)
# R sqaure value is 0.95 and p values are in the range
# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = start_up.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
start_up_train, start_up_test = train_test_split(start_up, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Profit ~ Research + Marketing', data = start_up_train).fit()

# prediction on test data set 
test_pred = model_train.predict(start_up_test)

# test residual values 
test_errors = test_pred - start_up_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_errors * test_errors))
test_rmse


# train_data prediction
train_pred = model_train.predict(start_up_train)

# train residual values 
train_resid  = train_pred - start_up_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


