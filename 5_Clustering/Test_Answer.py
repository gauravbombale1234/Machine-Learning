# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:09:46 2023

@author: Gaurav Bombale

Test File :Test_25_oct
"""

'''
Q.1 Determine if a letter is a vowel or a consonant
'''
V=['a','e','i','o','u']
letter=input("Enter the letter")

if letter in V:
    print("Letter is Vowel ")
else:
    print("Letter is Consonant")
    


'''
Q.2 Convert from a letter grade to a number of grade ponts
'''
dict={
A:4.0,
A_MINUS:3.7,
B_PLUS:3.3,
B:3.0,
B_MINUS:2.7,
C_PLUS:2.3,
C:2.0,
C_MINUS:1.7,
D_PLUS:1.3,
D:1.0,
F:0,
INVALID:-1
}

G=input("Enter the letter grade")
print(dict[G])
'''
Q 4 Draw inferences about the following boxplot & histogram.
Hint: [Insights drawn from the plots about the data such as whether data is normally 
distributed/not, outliers, measures like mean, median, mode, variance, std. deviation]
'''

# graph  is left Skewed






'''
Q 5 Below are the scores obtained by a student in tests 
34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
1) Find mean, median, variance, standard deviation.
2) What can we say about the student marks? [Hint: Looking at the various measures 
calculated above whether the data is normal/skewed or if outliers are present]. 
'''
import numpy as np
data=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]

print(np.mean(data))
print(np.median(data))
print(np.std(data))
'''
#The mean is 41 and median is 40.5 from that we can say that the graph is
slightly right skewed
#There is a clear outlier in the data, which is the score of 56. 
due to the mean is 41 and median is 40.5 and 56 is a very large 
value from the mean & median

'''


'''
Q.6
Calculate Mean, Median, Mode, Variance, Standard Deviation, Range & comment about the 
values / draw inferences, for the given dataset
- For Points, Score, Weigh>
Find Mean, Median, Mode, Variance, Standard Deviation, and Range and comment about the 
values/ Draw some inferences
'''
import pandas as pd

d=pd.read_excel("C:/Datasets/Assignment_module02.xlsx")

df=pd.DataFrame(d)
df.describe()

# MEAN
np.mean(df['Points'])   #Out[22]: 3.5965625
np.mean(df['Score'])    #Out[24]: 3.2115625000000003
np.mean(df['Weigh'])    #Out[25]: 17.848750000000003

#MEDIAN
np.median(df['Points']) #Out[26]: 3.6950000000000003
np.median(df['Score'])  #Out[27]: 3.325 
np.median(df['Weigh'])  #Out[28]: 17.71

#Standard Deviation
np.std(df['Points']) #Out[29]: 0.5262580722361891 
np.std(df['Score'])  #Out[30]: 0.9504535081179669 
np.std(df['Weigh'])  #Out[31]: 1.758800638929836


## Conclusion:
    '''
    from the above mean, median, std we can say that
    1) the values of mean and std are nearly equal. 
    2) 
    '''



