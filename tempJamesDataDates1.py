# -*- coding: utf-8 -*-

#####james
"""
Created on Thu Oct 27 03:15:35 2016

@author: space_000
"""

"""
Created on Wed Oct 26 21:59:51 2016

@author: Luke
"""

from sklearn import linear_model as skl
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import time, datetime

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0
    
#version for multiple types    
    
def unix_time_days(dt):
    return (dt - epoch).total_seconds() / 86400.0

def log_loss_binary(actual, predicted, eps = 1e-15):
    predicted = np.minimum(np.maximum(predicted, eps), 1 - eps)
    
    return -(sum(actual * np.log(predicted) + (1. - actual) * np.log(1. - predicted))) / len(actual)

train = pd.read_csv('C:/Users/space_000/qubHackathon/the_winning_model/trainingData.csv')
test = pd.read_csv('C:/Users/space_000/qubHackathon/the_winning_model/testingData.csv')

X = train[['garbage', 'rodents', 'sanitation', 'burglary', 'pastFail',
           'pastResponse', 'risk', 'timeSinceLast', 'ageAtInspection', 
           'consumptionOnPremises', 'inspectionDate', 'licenseStartDate',
           'expirationDate'
           ]]

#each date is in str
#str -> datetime -> numerical (days etc)

print "*******"
#print X['inspectionDate'][0]   for the value of a cell

#X['inspectionDate'] = pd.to_datetime(X['inspectionDate'])
#print unix_time_days(X['inspectionDate'][0])
#X['inspectionDate'] = unix_time_days(X['inspectionDate'])

#~~~~~~~~~~~~
def convertDateToDays(column):
    X[column + 'Days'] = pd.to_datetime(X[column])
    X[column + 'Days'] = X[column + 'Days'].astype(np.int64)
    X[column + 'Days'] = X[column + 'Days'].astype(np.float64)
    X[column + 'Days'] = X[column + 'Days'] / (1000000000.0 * 60.0**2 * 24.0)
    
X['ageAtInspectionDays'] = X['ageAtInspection'] * 365.0
X['timeSinceLastDays'] = X['timeSinceLast'] * 365.0    
    
convertDateToDays('inspectionDate')
convertDateToDays('licenseStartDate')
convertDateToDays('expirationDate')
print X['inspectionDateDays']
print X['licenseStartDateDays']
print X['expirationDateDays']

print X['timeSinceLastDays']
print X['ageAtInspectionDays']

#Data['G_ord'] = pd.Categorical(Data.G).labels
#print Data
X['constant'] = 1
riskdummies = pd.get_dummies(X['risk'], prefix='risk')

X = X.join(riskdummies)
del X['risk_Risk 1 (High)']
del X['risk']

#X['inspectionDate'] = pd.timestamp(X['inspectionDate'])
X['inspectionDate'] = pd.to_datetime(X['inspectionDate'])
X['inspectionDate'] = X['inspectionDate'].dt.month
insdatedummies = pd.get_dummies(X['inspectionDate'], prefix='insdate')
X = X.join(insdatedummies)
del X['insdate_1']
del X['inspectionDate']

X['licenseStartDate'] = pd.to_datetime(X['licenseStartDate'])
X['licenseStartDate'] = X['licenseStartDate'].dt.month
licdatedummies = pd.get_dummies(X['licenseStartDate'], prefix='licSdate')
X = X.join(licdatedummies)
del X['licenseStartDate']
del X['licSdate_1']

X['expirationDate'] = pd.to_datetime(X['expirationDate'])
X['expirationDate'] = X['expirationDate'].dt.month
licdatedummies = pd.get_dummies(X['expirationDate'], prefix='expdate')
X = X.join(licdatedummies)
del X['expirationDate']
del X['expdate_1']


#insTypedummies = pd.get_dummies(X['inspectionType'], prefix='insType')
#print insTypedummies
#####################

y = train['response']

logistic_model = skl.LogisticRegression()
logistic_model.fit(X, y)

training_predictions = logistic_model.predict_proba(X)
log_loss_binary(y, training_predictions[:, 1])
response_mean = np.array([y.mean()] * len(y))

print logistic_model.intercept_, logistic_model.coef_

multiple_predictors = logistic_model.predict_proba(X)

print str(log_loss_binary(y, response_mean)) + ' Log loss of response mean model'
print str(log_loss_binary(y, training_predictions[:, 1])) + ' Log loss of single predictor model' 
print str(log_loss_binary(y, multiple_predictors[:, 1])) + ' Log loss of multiple predictor model'
