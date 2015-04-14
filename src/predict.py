#!/usr/bin/env python
import csv
import sys
import math
import numpy as np
from sklearn import svm

# fit a model
def fit(X, y):
    print 'fitting the model...'
    
    svr = svm.SVR(C=4e6, degree=3, gamma=0.1)
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

def k_folds(X, y, k = 5, skip_cols=0):
    size = len(X) / k
    col = len(X[0])

    # contact it self  
    X = X + X
    y = y + y

    total_score = 0.0
    XX = np.array(X)
    yy = np.array(y)

    for i in range(k):
        idx_train_s = i*size
        idx_train_e = (i+k-1)*size
        idx_test_s  = (i+k-1)*size
        idx_test_e  = (i+k)*size

        print idx_train_s, idx_train_e, idx_test_e

        X_train = XX[idx_train_s:idx_train_e, :].tolist()
        y_train = yy[idx_train_s:idx_train_e].tolist()

        X_test = XX[idx_test_s:idx_test_e, :].tolist()
        y_test = yy[idx_test_s:idx_test_e].tolist()

        
        svr = fit(X_train, y_train)

        y_predict = predict(svr, X_test)

        score = evaluate(y_predict, y_test)

        total_score += score

    total_score = total_score / k

    return total_score


if __name__ == '__main__':
    train_filename = '../data/train_cleaned.csv'
    test_filename = '../data/test_cleaned.csv'

    skip_cols = 4

    train_data = np.array(read_csv(train_filename))
    
    X_train = train_data[:, skip_cols:len(train_data[0])-1].tolist()
    y_train = train_data[:, len(train_data[0])-1].tolist()
    
    avg_score = k_folds(X_train, y_train)

    print avg_score

    ################################################

    if len(sys.argv) >= 2 and sys.argv[1] == '-t':

        # rebuild the model
        svr = fit(X_train, y_train)

        X_test = np.array(read_csv(test_filename))
        X_test = X_test[:, skip_cols:len(train_data[0])].tolist()

        solution_filename = 'solution.csv'

        y_test_predict = predict(svr, X_test)

        write_solution(solution_filename, y_test_predict)