#!/usr/bin/env python
import csv
import sys
import math
import numpy as np
import smote
from sklearn import svm
from sklearn import cross_validation
from sklearn import kernel_ridge
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import *


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
produce model 
X: record, n*m matrix
Y: revenue, n*1 matrix
return: models
'''
def fit(X, y, config = {}):

    models = []

    if 'ensemble' in config:
        models = [
            #KNeighborsRegressor(n_neighbors=23, weights='distance'),
            # KNeighborsRegressor(n_neighbors=22, weights='distance'),
            #KNeighborsRegressor(n_neighbors=20, weights = 'distance'),
            #KNeighborsRegressor(n_neighbors=15, weights = 'distance'),   
            
            ########################################################################## 
            # for standardized data
            KNeighborsRegressor(n_neighbors=21, weights = 'distance'),
            # svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047),
            svm.NuSVR(nu=0.33, C=9e6, degree=2, gamma=0.0092),
            GradientBoostingRegressor(n_estimators=200, learning_rate=0.62, max_depth=1, random_state=1, loss='lad')
            ###########################################################################

            ########################################################################## 
            # for normalized data
            # KNeighborsRegressor(n_neighbors=21, weights = 'distance'),
            # svm.NuSVR(nu=0.35, C=5e7, degree=2, gamma=0.008),
            # GradientBoostingRegressor(n_estimators=150, learning_rate=0.8, max_depth=1, random_state=1, loss='lad')
            ###########################################################################

            # RandomForestRegressor(n_estimators=100, n_jobs=-1)
            #AdaBoostRegressor(n_estimators=100,  learning_rate = 0.3, loss='exponential')
            #GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=1, loss='lad'),
            #GradientBoostingRegressor(n_estimators=150, learning_rate=0.2, max_depth=3, random_state=1, loss='lad')
            #svm.SVR(C=1.3, degree=3, gamma=0.05)
        ]
    else:

        # model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.62, max_depth=1, random_state=1, loss='lad')

        # model =  RandomForestRegressor(n_estimators=200, n_jobs=-1)

        # model = BaggingRegressor(n_estimators=50, n_jobs=-1)

        # model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        model = svm.NuSVR(nu=0.33, C=9e6, degree=2, gamma=0.0092)
        # model = svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047)

        #model = RadiusNeighborsRegressor(radius=15, weights='distance')

        # model = AdaBoostRegressor(n_estimators=500,  learning_rate = 0.3, loss='exponential')

        # model = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0, loss='lad') 

        # score: 4.975
        # model = kernel_ridge.KernelRidge(alpha=50, kernel='linear')

        
        # model = svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047)

        # score: 3.19637548788
        # model = SGDClassifier(loss="hinge", penalty="l2")

        # score: 2.43654929108 weights = uniform
        # score: 2.37613424202 n_neighbors=20, weights = distance, test: 1730840.19422
        # model = KNeighborsRegressor(n_neighbors=21, weights = 'distance')

        # score: 3.40178270362
        # model = DecisionTreeRegressor()

        # score: 2.54716252715
        # model = svm.SVR(C=1.3, degree=3, gamma=0.05)

        models = [ model ]

    # oversampling the training set
    if 'oversampling' in config:
        X, y = smote.SMOTE(X, y, 200, 20)

    for model in models:
        print 'fitting model...'
        model.fit(X, y)

    return models

'''
predict by given models
models: given models
X: record to predict, n*m matrix
return: predicted (mean) revenue, n*1 matrix
'''
def predict(models, X):

    ys = []

    for model in models:
        print 'predicting...'    
        ys.append(model.predict(X))

    # return mean prediction
    y = np.mean(np.array(ys), axis=0)

    return y


'''
k fold validation
return: average Root Mean Squared Eroor (RMSE)
'''
def k_folds(X, y, k = 5, config={}):
    n = len(X)

    shuffle = True if 'shuffle' in config else False

    kf = cross_validation.KFold(n=n, n_folds=k, shuffle=shuffle)

    total_score = 0.0

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        models = fit(X_train, y_train, config)
        y_predict = predict(models, X_test)

        score = math.sqrt(mean_squared_error(y_predict, y_test))

        total_score += score

        print 'score =', score / 1e6        

    total_score /= k

    return total_score

def predict_test(X_train, y_train, X_test, config):
    
    # rebuild the model

    models = fit(X_train, y_train, config)
    y_predict = predict(models, X_test)

    solution_filename = 'solution.csv'

    write_solution(solution_filename, y_predict)

def parse_arg(argv):
    config = {}
    config['date'] = True
    config['city_group'] = True
    config['city_name'] = True
    config['type'] = True

    for arg in argv:
        if arg == '-t':
            config['test'] = True
        if arg == '-pca':
            config['pca'] = True
        if arg == '-os':
            config['oversampling'] = True
        if arg == '--no-date':
            config.pop('date', None)
        if arg == '--no-city-group':
            config.pop('city_group', None)
        if arg == '-e':
            config['ensemble'] = True
        if arg == '-s':
            config['shuffle'] = True
        if arg == '-r':
            config['repeat'] = True
            config['shuffle'] = True
        if arg == '--no-city-name':
            config.pop('city_name', None)
        if arg == '--no-type':
            config.pop('type', None)
    return config

if __name__ == '__main__':

    config = parse_arg(sys.argv)

    train_filename = '../data/train_cleaned.csv'
    test_filename = '../data/test_cleaned.csv'

    if 'pca' in config:
        train_filename = '../data/train_pca.csv'
        test_filename = '../data/test_pca.csv'

    # read training data and convert to numpy array
    train_data = np.array(read_csv(train_filename), dtype=float)

    if 'pca' in config:
        # if use pca, use all features
        cols = [i for i in range(0, len(train_data[0])-1)]
    else:        

        # selected features
        cols = [] 
        
        if 'date' in config: cols = cols + [0]
        if 'city_name' in config: cols = cols + [1]
        if 'city_group' in config: cols = cols + [2]
        if 'type' in config: cols = cols + [3]

        cols = cols + [i for i in range(4, len(train_data[0])-1)]
    
    print len(cols)
    
    
    # split training data to (record) and (predicted value)
    X_train = train_data[:, cols]
    y_train = train_data[:, len(train_data[0])-1]
    
    # calcualte average Root Mean Squared Eror (RMSE)
    total_avg = 0.0

    repeat = 10 if 'repeat' in config else 1

    for i in range(repeat):
        print '----- repeat %s ----' % i
        avg_score = k_folds(X_train, y_train, k=5, config=config)
        total_avg += avg_score

    total_avg /= repeat

    print total_avg / 1e6

    ################################################
    # run test

    if 'test' in config:
        X_test = np.array(read_csv(test_filename), dtype=float)
        X_test = X_test[:, cols]
        predict_test(X_train, y_train, X_test, config)
