# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:42:36 2024

@author: Gaurav Bombale
"""
"""
Sales of products in four different regions is tabulated for males and
females. Find if male-female buyer rations are similar across regions.
"""
import pandas as pd
import numpy as np
import scipy
from scipy import stats
sales=pd.read_csv("BuyerRatio.csv")
#Here input data is dicrecrete and output is also discrete
#samples are more than 2,hence chi_square test need to be applied
sales_table=sales.iloc[:,1:6]
sales_table
sales.head()
sales_table.values
#H0=male-female buyer relations are similar across regions.
#H1=male-female buyers relations are different
Chisquares_result=scipy.stats.chi2_contingency(sales_table.values)
Chisquares_result
Chi_square=[['Test statistics','p-value'],Chisquares_result[0],Chisquares_result[1]]
Chi_square
#p-value=0.6603094907091882>0.05 ,p high null fly,H0 is true
#male-female buyer relations are similar across regions.
