# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:33:41 2024

@author: Gaurav Bombale
"""

import pandas as pd
import  numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
#from statsmodel.stats import Weightstats as stests
import statsmodels.stats.weightstats
# 1 sample sign test
# for given dataset check whether  scores are equal or less than 80
# H0=scores  are either equal or less than 80
# H1=scores are not equal n greater than 80
# Whenever there is single sample  and data is not normal

########### 1 sample sign Test #########
# students scores data
marks=pd.read_csv("C:/12_Hypothesis Testing/hypothesis_datasets/Signtest.csv")
#Normal QQ plot 
import pylab
stats.probplot(marks.Scores,dist='norm',plot=pylab)
#Data is not normal
# H0 - Data is  normal
# H1 - Data is not normal
stats.shapiro(marks.Scores)
#p_value is 0.024 > 0.005, p is high null fly 
#Decision : data is  not normal
###########
#let us check the distribution of the data 
marks.Scores.describe()
#1 sample sign test

import matplotlib.pyplot as plt 
plt.boxplot(marks.Scores)

# Non-Parametric Test--> 1 sample sign test






###### 1 sample Z test ############
#importing the data
fabric=pd.read_csv("C:/12_Hypothesis Testing/hypothesis_datasets/Fabric_data.csv")

#calculating the normality test
print(stats.shapiro(fabric))
#0.1450 > 0.005 means H0 is True
#calculating the mean
np.mean(fabric)

# Z test
#parameters in z test , value is  mean of data
ztest, pval=stests.ztest(fabric, x2=None, value=158)

print(float(pval))

# p-value = 7.156e-06 < 0.05 so p low null go


#########################################
####### Man Whitsney Test 
#Vehicle  with and without addictive
# H0 : fuel additive does not impact the performance
# H1 : fuel additive does impact the perforance

fuel=pd.read_csv("C:/12_Hypothesis Testing/hypothesis_datasets/mann_whitney_additive.csv")
fuel 

fuel.columns="Without_additive","With_additive"
#Normality test 
# H0 : data is normal
print(stats.shapiro(fuel.Without_additive))   # p high null fly
print(stats.shapiro(fuel.With_additive))      #p low null go
#without_additive is normal
#with additive is not normal
#when two sampls are not normal then manwhitney test
#Non-Parameteric Test case
# Man Whitney test
scipy.stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)

# p-value  = 0.4457 so p high null fly 
# H0 : fuel addition does not impact the performance




##############################################
######## Paired T - Test
#when two feature are normal then paired T test
#A univariate test that tests for a significant difference between 2 relation

sup=pd.read_csv("C:/12_Hypothesis Testing/hypothesis_datasets/paired2.csv")
sup.describe()

# H0 : There is no significant difference between means of supplier of A and B
# Ha : There is significant difference between means of suppliers of A and B
# Normality Test - Shapiro Test
stats.shapiro(sup.SupplierA)

stats.shapiro(sup.SupplierB)
#Data are normal

import seaborn as sns
sns.boxplot(data=sup)

#Assuming the external Conditions are same for both the samples
#Paired T-test
ttest, pval = stats.ttest_rel(sup["SupplierA"], sup["SupplierB"])
print(pval)

# pval = 0 < 0.05 so p low  null go


#####################################
############# 2 sample T-Test 
# load the data
prom=pd.read_excel("C:/12_Hypothesis Testing/hypothesis_datasets/Promotion.xlsx")
prom
# H0 : InterestRateWaiver < StandardPromotion
# Ha : InterestRateWaiver > StandardPromotion
prom.columns="InterestRateWaiver","StandardPromotion"
#Normality test 
stats.shapiro(prom.InterestRateWaiver)  # Shapiro Test

print(stats.shapiro(prom.StandardPromotion))

#Data are normal

#Variance Test
help(scipy.stats.levene)
# H0: Both Columns have equal variance
# H1: Both Columns have unequal variance

scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
#p-value=0.287 > 0.05 so p high null fly => Equal variance

##2 sample T test
scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.ttest_ind)

###################################
######## One Way ANOVA
'''
A marketing organization outsources their back-office operations
to three different supppliers. The contract are up for renewal and 
the CMO wants to determine whether they should renew contract with all
suppliers or any specific supplier. CMO want to renew thw contract of suppliers ........
'''
con_renewal=pd.read_excel("C:/12_Hypothesis Testing/hypothesis_datasets/ContractRenewal_Data(unstacked).xlsx")
con_renewal
con_renewal.columns="SupplierA","SupplierB","SupplierC"

# H0: All the 3 suppliers have equal mean transaction time
# H1: All the 3 suppliers have not equal mean transaction time

# Normality Test
stats.shapiro(con_renewal.SupplierA) # Shapiro Test
# pvalue=0.89> 0.005 SupplierA is Normal
stats.shapiro(con_renewal.SupplierB) # Shapiro Test

stats.shapiro(con_renewal.SupplierC) # Shapiro Test
# pvalue=0.57> 0.005 SupplierC is Normal

#Variance Test
help(scipy.stats.levene)
#All 3 suppliers are being checked for variances
scipy.stats.levene(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)
#The Levene test tests the null hypothesis 
# that all input samples are from populations with equal variance
# pvalue=0.777>0.005 p is high  null fly 
# H0: All input Samples are from populations with equal variance

#One Way ANOVA
F,p=stats.f_oneway(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)

# p value
p  #  p high  Null fly
# all the 3 suppliers have equal mean transaction time



###################################
#########  2 Proportion Test    ###########
'''
Johnnie Talkers soft drinks division sales manager has been planning to 
launch a new sales incentive program for their sales executives. The sales
executives felt that adults (>40 yrs) wont buy , children will & hence 
requested sales manager not to launch the program . Anlayze...

'''
import numpy as np

two_prop_test=pd.read_excel("C:/12_Hypothesis Testing/hypothesis_datasets/JohnyTalkers.xlsx")

from statsmodels.stats.proportion import proportions_ztest

tab1=two_prop_test.Person.value_counts()
tab1
tab2=two_prop_test.Drinks.value_counts()
tab2

#crosstable table 
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

count=np.array([58,152])
nobs=np.array([480,740])

stats, pval = proportions_ztest(count, nobs, alternative="two-sided")
print(pval)  # Pvalue= 0.000

stats, pval = proportions_ztest(count, nobs, alternative="larger")
print(pval)  # Pvalue= 0.999



#######################################
#######  Chi-Square Test      ###########
Bahaman=pd.read_excel("C:/12_Hypothesis Testing/hypothesis_datasets/Bahaman.xlsx")
Bahaman 

count = pd.crosstab(Bahaman['Defective'],Bahaman['Country'])
count
Chisquares_results =  scipy.stats.chi2_contingency(count)

Chi_square=[['Test Statistic','p-value'],[Chisquares_results[0],Chisquares_results[1]]]
Chi_square
'''
you use chi2_contingency when you want test 
whether two (or more ) groups have the same distribution
'''
#H0: Null Hypothesis: the two groups have no significant difference


