# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:42:36 2024

@author: Gaurav Bombale
"""
"""
Telecall uses 4 centers around the globe to process customer order forms.
They audit a certain % of the customer order forms. Any error in order form
renders it defective and must be reworked before processing. The manager
wants to check whether the defective % varies by center. Please analyze
the data at 5% significance level and help the manager draw appropriate
inferences
"""
import pandas as pd
import numpy as np
import scipy
from scipy import stats
data=pd.read_csv("CustomerOrderform.csv")
#Here input data is dicrecrete and output is also discrete
#samples are more than 2,hence chi_square test need to be applied
#sales_table=sales.iloc[:,1:6]
#sales_table
data.head()
data.Phillippines.value_counts()
data.Indonesia.value_counts()
data.Malta.value_counts()
data.India.value_counts()

# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs
# Chi2 contengency independence test
chi_sq=scipy.stats.chi2_contingency(obs)
chi_sq
## o/p is (Chi2 stats value, p_value, df, expected obsvations)
#p_value=0.27710209912>0.05,H0 is True
#H0=There are no errors across countries.
#H1=There are  errors across countries.

