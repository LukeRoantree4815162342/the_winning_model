# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:46:58 2016

@author: Luke
"""

import pandas as pd
from sklearn import linear_model 
import numpy as np
import statsmodels.formula.api as smf

def log_loss_binary(actual, predicted, eps = 1e-15):
    predicted = np.minimum(np.maximum(predicted, eps), 1 - eps)
    
    return -(sum(actual * np.log(predicted) + (1. - actual) * np.log(1. - predicted))) / len(actual)

train = pd.read_csv('C:/Users/Luke/QUBHACKATHON/trainingData.csv') #You'll need to make these your own paths
test = pd.read_csv('C:/Users/Luke/QUBHACKATHON/testingData.csv')


regressors = train[[train.columns.values[7], train.columns.values[8], train.columns.values[38], train.columns.values[37]]]

response= train['response']
est = smf.OLS(response, regressors)
est = est.fit()
print est.summary()
print type(est.summary())

regressors_new = test[[test.columns.values[7], test.columns.values[8], test.columns.values[38], test.columns.values[37]]]

print ((est.predict(regressors)*2.0).round())*1.0/1.0
print response.values

real = np.array(response.values)
predicted = np.array(((est.predict(regressors)*2.0).round())*1.0/1.0)

print np.average(real-predicted)
print '\n'
print log_loss_binary(real, predicted)

