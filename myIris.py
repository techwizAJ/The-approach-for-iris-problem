# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:49:14 2018

@author: techwiz
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#importing dataset
from sklearn.datasets import load_iris
orginalDataset = load_iris()
dataset = load_iris(return_X_y=True)
X = dataset[0]
y = dataset[1]
y = y.reshape(len(y),1)

#Splitting dataset into train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

#Traning Model
#Using RandomForest Classifier  to classify
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200,criterion ='entropy',random_state=0)
clf.fit(X_train,y_train)
# Using  KNN algorithm to classify
from sklearn.neighbors import KNeighborsClassifier
clfKNN = KNeighborsClassifier()
clfKNN.fit(X_train , y_train)

#Testing Model
y_pred = clf.predict(X_test)
y_predKNN = clfKNN.predict(X_test)

#Evaluating Model Based on confusion Matrix
from sklearn.metrics import confusion_matrix
cnf_RF = confusion_matrix(y_test,y_pred)
cnf_KNN = confusion_matrix(y_test,y_predKNN)

from sklearn.metrics import accuracy_score
acu_RF = accuracy_score(y_test,y_pred)
acu_KNN = accuracy_score(y_test,y_predKNN)