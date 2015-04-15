#!/usr/bin/env python
import csv
import sys
import math
import numpy as np
from sklearn import svm
from sklearn import kernel_ridge
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

'''
produce model 
X: record, n*m matrix
Y: revenue, n*1 matrix
return: model
'''
def fit(X, y):
    print 'fitting the model...'

    # score: 2.44629209987
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0, loss='lad')    
    
    # score: 4.975
    # model = kernel_ridge.KernelRidge(alpha=50, kernel='linear')

    # 2.36622251546 k = 5
    #model = svm.NuSVR(nu=0.25, C=1.5e7, degree=2, gamma=0.0042)

    # score: 3.19637548788
    # model = SGDClassifier(loss="hinge", penalty="l2")

    # score: 2.43654929108 weights = uniform
    # score: 2.37613424202 weights = distance, test: 1730840.19422
    model = KNeighborsRegressor(n_neighbors=20, weights='distance')

    # score: 3.40178270362
    # model = DecisionTreeRegressor()

    # score: 2.54716252715
    # model = svm.SVR(C=1.3, degree=3, gamma=0.05)

    model.fit(X, y)

    return model


'''
predict using a given model
model: given model
X: record to predict, n*m matrix
return: predicted revenue, n*1 matrix
'''
def predict(model, X):
    print 'predicting...'
    
    y = model.predict(X).tolist()

    return y

'''
read file, e.g., training file, testing file
return: n*m matrix
'''
def read_csv(filename):    
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()
        for row in reader:
            rows.append(row)
    return rows

'''
write solution file
format: id, predicted value
'''
def write_solution(filename, y):
    print 'writeing solution to %s' % filename
    with open(filename, 'w') as f:
        f.write('Id,Prediction\n')
        for i in range(len(y)):
            f.write(str(i)+','+str(y[i])+'\n')
    print 'done'


'''
k fold validation
return: average Root Mean Squared Eroor (RMSE)
'''
def k_folds(X, y, k = 5, skip_cols=0):
    size = len(X) / k
    col = len(X[0])

    # contact it self  
    X = X + X
    y = y + y

    total_score = 0.0
    XX = np.array(X, dtype=float)
    yy = np.array(y, dtype=float)

    for i in range(k):
        idx_train_s = i*size
        idx_train_e = (i+k-1)*size
        idx_test_s  = (i+k-1)*size
        idx_test_e  = (i+k)*size

        X_train = XX[idx_train_s:idx_train_e, :]
        y_train = yy[idx_train_s:idx_train_e]

        X_test = XX[idx_test_s:idx_test_e, :]
        y_test = yy[idx_test_s:idx_test_e]
        
        svr = fit(X_train, y_train)

        y_predict = predict(svr, X_test)

        score = math.sqrt(mean_squared_error(y_predict, y_test))

        print 'score =', score / 1e6

        total_score += score

    total_score = total_score / k

    return total_score

def predict_test(X_train, y_train, X_test):
    # rebuild the model
    svr = fit(X_train, y_train)

    y_predict = predict(svr, X_train)

    score = math.sqrt(mean_squared_error(y_predict, y_train))

    print 'train score =', score / 1e6    

    solution_filename = 'solution.csv'

    y_test_predict = predict(svr, X_test)

    write_solution(solution_filename, y_test_predict)


if __name__ == '__main__':
    train_filename = '../data/train_cleaned.csv'
    test_filename = '../data/test_cleaned.csv'

    # train_filename = '../data/train_pca.csv'
    # test_filename = '../data/test_pca.csv'

    # read training data and convert to numpy array
    train_data = np.array(read_csv(train_filename), dtype=float)

    # selected features open date, p1 ~ p37
    cols = [0] + [i for i in range(4, len(train_data[0])-1)]
    print len(cols)
    
    
    # split training data to (record) and (predicted value)
    X_train = train_data[:, cols].tolist()
    y_train = train_data[:, len(train_data[0])-1].tolist()
    
    # calcualte average Root Mean Squared Eror (RMSE)
    avg_score = k_folds(X_train, y_train, k=5)

    print avg_score / 1e6

    ################################################
    # run test

    if len(sys.argv) >= 2 and sys.argv[1] == '-t':
        X_test = np.array(read_csv(test_filename), dtype=float)
        X_test = X_test[:, cols]
        predict_test(X_train, y_train, X_test)