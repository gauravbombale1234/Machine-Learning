# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:42:35 2024

@author: Gaurav Bombale
"""
"""
A hospital wants to determine whether there is any difference in the
average Turn Around Time (TAT) of reports of the laboratories on their
preferred list. They collected a random sample and recorded TAT for
reports of 4 laboratories. TAT is defined as sample collected to report
dispatch.
Analyze the data and determine whether there is any difference in
average TAT among the different laboratories at 5% significance level.
"""
import pandas as pd
import numpy as np
import scipy
from scipy import stats
tat=pd.read_csv("lab_tat_updated.csv")
tat.head()
#There are four samples input is discrete,output is continious
tat.columns="Lab_A","Lab_B","Lab_C","Lab_D"
tat.head()
tat.isna().sum()
#Now let us check the normality 
#H0=data is normal
#H1=Data is not normal
stats.shapiro(tat.Lab_A)
#pvalue=0.4231 >0.05 hence data is normal
stats.shapiro(tat.Lab_B)
#pvalue=0.8637 >0.05 hence data is normal
stats.shapiro(tat.Lab_C)
#pvalue=0.0654 >0.05 hence data is normal
stats.shapiro(tat.Lab_D)
#pvalue=0.6618 >0.05 hence data is normal
#All the columns are normal
#Variance Test
#H0=variance between Lab_A=Lab_B=Lab_C=Lab_D is equal
#H1=At least one has got different variance
scipy.stats.levene(tat.Lab_A,tat.Lab_B,tat.Lab_C,tat.Lab_D)
#pvalue=0.3810>0.05 hence p high null fly,there is equal variance
#H0:All the labs are having equal TAT
#H1:At least one is having different TAT

F,pval=stats.f_oneway(tat.Lab_A,tat.Lab_B,tat.Lab_C,tat.Lab_D)
pval
#2.143740909435053e-58<0.05,p low null go,H1 is true,at least one
#lab has got different TAT
