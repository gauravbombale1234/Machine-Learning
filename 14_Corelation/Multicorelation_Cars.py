# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:59:38 2024

@author: Gaurav Bombale
"""
import pandas as pd
import numpy as np
import seaborn as sns

cars=pd.read_csv('Cars.csv')
#EDA
# 1. Measure the central tendancy
# 2. Measures of dispersion
# 3. Third moment business decision -skewness
# 4. Fourth moment business decision - kurtosis
# 5. probability distribution
# 6. Graphical representation(Histogram, Boxplot)
cars.describe()
#Graphical representation
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,81,1))
sns.distplot(cars.HP)
#data is right skewed
plt.boxplot(cars.HP)
#There are several outliers in HP columns
#Similar operation are expected for other three column
sns.distplot(cars.MPG)
#data is slightlt left distributed
plt.boxplot(cars.MPG)
#There is no outlier
sns.displot(cars.VOL)
#data is slightlt left distributed
plt.boxplot(cars.VOL)
sns.displot(cars.SP)
#Data is slightlt right distributed
plt.boxplot(cars.SP)
#There are several outliers
sns.displot(cars.WT)
plt.boxplot(cars.WT)
#There are several outliers

#Now let us plot joint plot , joint plot  is to show scatter histogram
sns.jointplot(x=cars['HP'],y=cars['MPG'])

#now let us plot count plot 
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times the each value occured
#92 HP value occured 7 times

#QQ plot 
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist='norm',plot=pylab)
plt.show()
#MPG dat is normally distributed 
#there are 10 scatter plots need to be plotted , one by one
#to plot , so we can use pair plot

sns.pairplot(cars.iloc[:,:])
# write  Linearity : , Direction:   strength
#you can check the collinearity problem between the input 
#you can check plot between SP and HP , they are strongly corelated
#same way you can check WT and VOL , it is also strongly corelated

#now let us check r value between variables
cars.corr()
#you can check SP and HP  ,r value is 0.97 and same way
#you can  check WT ad VOL , it has got 0.999
# which is greater 
#now although we observed strongly corelated pairs, #
#still we will go for Linear Regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
#R square value observed is 0.771 < 0.85
#p-values of WT and VOL is 0.814 and 0.556 which is very high
#it means it is greater than 0.05 , WT and VOL columns 
#we need to ignore it 
#or delete . Instead deleting 81 entries,
#let us check row wise outliers
#identifying is there any influential value.
#To check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers
#go to data frame and check 76th entry 
#let us delete that entry 
cars_new=cars.drop(cars.index[[76]])

#again apply regression to cars new
ml_new=smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()
#R-square value is 0.819 but p values are same , hence not 
#Now next option is delete the column but  question is which columns
#is to be deleted. We have already checked correlation factor r
#VOL has got -0.529 and for WT =-0.526
#WT is less hence can be deleted

#another apporoach is to check the collinearity 
#rsquare is giving that value,
#we will have to apply  regression w.r.t x1 and input
#as x2,x3 ad x4 so and so forth 
rsq_hp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp
#VIF is variance  influential factor , calculating VIF helps
#of x1 w.r.t. x2,x3 and x4
rsq_wt=smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_wt)

rsq_vol=smf.ols('VOL~HP+WT+SP',data=cars).fit().rsquare
vif_vol=1/(1-rsq_vol)

rsq_sp=smf.ols('SP~HP+WT+VOL',data=cars).fit().rsquared
vif_sp=1/(1-rsq_sp)

#vif_wt=639.53 and vif_vol=638.80 hence vif_wt
#is greater , thumb rule is vif should not to be greater than 1

#storing the values in dataframe 
d1={'Varibles':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame

### let us drop WT and apply correlation to remailing three 
final_ml=smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
#R-Square is 0.770 and p values 0.00,0.012 < 0.05


#prediction
pred=final_ml.predict(cars)

##QQ plot 
res=final_ml.resid
sm.qqplot(res)
plt.show()
#This QQ plot is on residual which is obtained on training 
#errors are obtained on test data
stats.probplot(res,dist='norm',plot=pylab)
plt.show()

#let us plot the residual plot , which takes the residuals
#and the data
sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS  Residual')
plt.show()
#residual plots are used to check 

# spliting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test=train_test_split(cars,test_size=0.2)
# preparing the model on train data
model_train=smf.ols('MPG~VOL+SP+HP',data=cars_train).fit()
model_train.summary()
test_pred=model_train.predict(cars_test)

test_error=model_train.predict(cars_test)
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse