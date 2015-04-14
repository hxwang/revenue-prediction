#!/usr/bin/env python
import csv
import sys
import math
import numpy as np
from sklearn import svm

# fit a model
def fit(X, y):
    print 'fitting the model...'
    
    svr = svm.SVR()
    svr.fit(X, y)

    return svr

# predict based a model
def predict(model, X):
    print 'predicting...'
    
    y = model.predict(X).tolist()

    return y

# evaluate the result
def evaluate(y_predict, y_truth):

    print 'evaluating...'

    rmse = 0
    for i in range(len(y_predict)):
        rmse = rmse + (float(y_predict[i]) - float(y_truth[i]))**2

    rmse = rmse / len(y_predict)

    rmse = math.sqrt(rmse)

    return rmse

# read a csv file into rows
def read_csv(filename):    
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')     
        for row in reader:
            rows.append(row)
    return rows

# write the preidtion into solution file
def write_solution(filename, y):
    print 'writeing solution to %s' % filename
    with open(filename, 'w') as f:
        f.write('Id,Prediction\n')
        for i in range(len(y)):
            f.write(str(i)+','+str(y[i])+'\n')
    print 'done'



if __name__ == '__main__':
    train_filename = '../data/train_cleaned.csv'
    test_filename = '../data/test_cleaned.csv'

    train_data = np.array(read_csv(train_filename))
    
    X_train = train_data[:, 0:len(train_data[0])-1].tolist()
    y_train = train_data[:, len(train_data[0])-1].tolist()
    
    svr = fit(X_train, y_train)
    y_train_predict = predict(svr, X_train)

    print y_train
    print y_train_predict


    score = evaluate(y_train_predict, y_train)

    print score

    #################################################

    X_test = read_csv(test_filename)

    solution_filename = 'solution.csv'

    y_test_predict = predict(svr, X_test)

    write_solution(solution_filename, y_test_predict)