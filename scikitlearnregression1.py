# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 21:59:51 2016

@author: Luke
"""

from sklearn import linear_model as skl
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import time, datetime

def log_loss_binary(actual, predicted, eps = 1e-15):
    predicted = np.minimum(np.maximum(predicted, eps), 1 - eps)
    
    return -(sum(actual * np.log(predicted) + (1. - actual) * np.log(1. - predicted))) / len(actual)

train = pd.read_csv('C:/Users/Luke/QUBHACKATHON/trainingData.csv') #You'll need to make these your own paths
test = pd.read_csv('C:/Users/Luke/QUBHACKATHON/testingData.csv')
def sanitise(data):
    X = data[['garbage', 'rodents', 'sanitation', 'burglary', 'pastFail',
               'pastResponse', 'risk', 'timeSinceLast', 'ageAtInspection', 
               'consumptionOnPremises', 'inspectionDate', 'licenseStartDate',
               'expirationDate'
               ]]
               
    X['constant'] = 1
    riskdummies = pd.get_dummies(X['risk'], prefix='risk')
    
    X = X.join(riskdummies)
    del X['risk_Risk 1 (High)']
    del X['risk']
    
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
    return X
def getModel():
    X = sanitise(train)
    y = train['response']
    
    logistic_model = linear_model.LogisticRegression()
    logistic_model.fit(X, y)
    
    training_predictions = logistic_model.predict_proba(X)
    log_loss_binary(y, training_predictions[:, 1])
    response_mean = np.array([y.mean()] * len(y))
    
    #print logistic_model.intercept_, logistic_model.coef_
    
    multiple_predictors = logistic_model.predict_proba(X)
    #return logistic_model.intercept_, logistic_model.coef_
    return logistic_model
def predictme(data):
    X = sanitise(data)
    formula = getModel().predict_proba(X)
    #print len(formula[0])
    predictions = np.zeros(X.shape[1])
    index = 0
    predictions += getModel().intercept_
    for i in X.columns:
        #predictions += X[i]*formula[0][index]
       # print formula[0][index]
        #print X[i]*formula[0][index]
        index+=1
    return formula
def column(matrix, i):
    return [row[i] for row in matrix]
#print sanitise(test)
#print str(log_loss_binary(y, response_mean)) + ' Log loss of response mean model'
#print str(log_loss_binary(y, training_predictions[:, 1])) + ' Log loss of single predictor model' 
#print str(log_loss_binary(y, multiple_predictors[:, 1])) + ' Log loss of multiple predictor model'
predictions = predictme(test)
#
#for i in range(len(predictions)):
#    predictions[i] = predictions[i][0]
predictions = column(predictions, 0)
output_dataset = pd.DataFrame({'inspectionId': test.inspectionId,
                               'response': predictions})
output_dataset.head()
output_dataset.to_csv('submissionOne.csv', index = False)


