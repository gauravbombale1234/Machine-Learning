# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:17:40 2024

@author: Gaurav Bombale
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
#conda install mixtend
from mlxtend.classifier import StackingClassifier
import warnings
from sklearn import datasets
iris=datasets.load_iris()
X_train,y_train=iris.data[:,1:3],iris.target
weak_l1=KNeighborsClassifier(n_neighbors=1)
weak_l2=RandomForestClassifier(random_state=1)
weak_l3=GaussianNB()

###
meta_l=LogisticRegression()
stackingclf=StackingClassifier(classifiers=[weak_l1,weak_l2,weak_l3],meta_classifier=meta_l)

####
print("After three fold cross validation ")
for iterclf,iterlable in zip([weak_l1,weak_l2,weak_l3,stackingclf],['k-nearest Neighbour Model','Random Forest Model','Naive Bayes Model','Stacking classifier model']):
    scores=model_selection.cross_val_score(iterclf, X_train,y_train,cv=3,scoring='accuracy')
    print('Accuracy: ',scores.mean(),' for ',iterlable)



