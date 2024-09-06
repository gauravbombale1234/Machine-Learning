# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:42:35 2024

@author: Gaurav Bombale
"""

"""
A F&B manager wants to determine whether there is any significant
difference in the diameter of the cutlet between two units. A randomly
selected sample of cutlets was collected from both units and measured?
Analyze the data and draw inferences at 5% significance level. Please state
the assumptions and tests that you carried out to check validity of the
assumptions.
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
cutlet=pd.read_csv("Cutlets.csv")
cutlet.head()
#Here two samples are having input X is dicrete and diameter is continous
cutlet.columns="Unit_A","Unit_B"

cutlet.head()
cutlet.Unit_A.isna().sum()
#There are 16 na values
cutlet.Unit_B.isna().sum()
#There are 16 na values
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

cutlet['Unit_A']=pd.DataFrame(mean_imputer.fit_transform(cutlet[['Unit_A']]))
cutlet.Unit_A.isna().sum()
cutlet['Unit_B']=pd.DataFrame(mean_imputer.fit_transform(cutlet[['Unit_B']]))
cutlet.Unit_A.isna().sum()

#let us check the normality of two samples
#H0=data is normal
#H1=data is not normal
print(stats.shapiro(cutlet.Unit_A))
#pvalue=0.07343>0.05, p is high null fly,hence data is normal
print(stats.shapiro(cutlet.Unit_B))
#pvalue=0.017<0.05, p is low,null go hence data is not normal
#Now let us apply Mann-Whitney Test
#H0=Diameters of cutlets are same
#H1=Diameters of cutlets are different 
scipy.stats.mannwhitneyu(cutlet.Unit_A,cutlet.Unit_B)
#pvalue=0.1790>0.05,p is high null fly,H0 is true
#Diameter of cutlets are same
