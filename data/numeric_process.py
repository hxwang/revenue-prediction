#!/usr/bin/env python

import csv
import sys
import time
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, Imputer
import numpy as np
from read_data import *






def imputation(X_all, n_train, y_train, train_header, test_header):
    pass

def parse_arg(args):
    config = {}
    for arg in args:
        if arg[0] == '-': config[arg]=True

    return config


if __name__ == '__main__':
    train = 'train.csv'
    test = 'test.csv'

    config = parse_arg(sys.argv)

    # read data
    rows, rows_train, rows_test = get_all_rows(train, test)
    city_names = scan_city_name(rows)
    city_groups = scan_city_group(rows)
    types = scan_type(rows)

    print 'citys = %s, %s ' % (len(city_names), city_names)
    print 'city group = %s, %s ' % (len(city_groups), city_groups)
    print 'types = %s, %s ' % (len(types), types)

    train_header = read_csv_header(train)
    test_header = read_csv_header(test)
    train_header = train_header[1:len(train_header)]
    test_header = test_header[1:len(test_header)]

    # tranfrom data to numeric types
    rows_train = transform_data(rows_train, city_names, city_groups, types)
    rows_test = transform_data(rows_test, city_names, city_groups, types)

    # convert to float type
    rows_train = np.array(rows_train, dtype = float)
    rows_test = np.array(rows_test, dtype = float)
    # print 'rows_test.shapre', rows_test.shape    

    # write_csv('train_raw.csv', rows_train, train_header, True)
    # write_csv('test_raw.csv', rows_test, test_header, True)

    # merge train and test in order to do processing together
    X_train = rows_train[:, 1:len(rows_train[0])-1]
    y_train = rows_train[:, len(rows_train[0])-1]
    X_test  = rows_test[:, 1:len(rows_test[0])]

    print 'X_train.shape =', X_train.shape
    print 'y_train.shape =', y_train.shape
    print 'X_test.shape =', X_test.shape

    X_all   = np.vstack((X_train, X_test))
    
    n_train = len(X_train)

if '-s' in config:
    print 'Standardizing...'
    
    # standization, mean = 0, variance = 1
    X_all = scale(X_all)

    # split trainning and test data
    X_train = X_all[0:n_train,:]
    X_test  = X_all[n_train:len(X_all),:]
    # print 'X_test.shapre', X_test.shape
    
    # put revenue back
    Xy_train = np.c_[X_train, y_train]

    write_csv('train_scaled.csv', Xy_train, train_header, False)
    write_csv('test_scaled.csv', X_test, test_header, False)

# drop categorical feature
# city-name, type
if '-i' in config or '-pt' in config or '-ps' in config:
    
    # 'open date' and 'city groups' and 'p1-p37'
    cols = [0, 2] + [i for i in range(4, 4+37)]

    print len(cols)

    # drop off other feaatures
    X_all = X_all[:, cols]
    # X_train = X_train[:, cols]
    # X_test = X_test[:, cols]

if '-pt' in config or '-ps' in config:
    print 'PCA...'

    pca = PCA(n_components=15)    
    
    X_all = scale(X_all)

    print X_all.shape

    if '-pt' in config:            
        X_all = pca.fit_transform(X_all)
        # split trainning and test data
        X_train = X_all[0:n_train,:]
        X_test  = X_all[n_train:len(X_all),:]
    if '-ps' in config:
        # split trainning and test data
        X_train = X_all[0:n_train,:]
        X_test  = X_all[n_train:len(X_all),:]

        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    print 'variance = ', pca.explained_variance_ratio_
    print 'total variance = ', np.sum(pca.explained_variance_ratio_)
    
    # put revenue back
    Xy_train = np.c_[X_train, y_train]

    pca_headers = ['P' + str(i) for i in range(1, len(X_train[0])+1)]
    
    write_csv('train_pca.csv', Xy_train, pca_headers + ['revenue'], False)
    write_csv('test_pca.csv', X_test, pca_headers, False)

if '-i' in config: 

    print 'Imputating...'
    # make city groups has value 1 and 2 instead 0 and 1...
    X_all[:,1] += 1

    # create imputer object
    imputer = Imputer(missing_values=0, strategy='mean', verbose=1)

    X_all = imputer.fit_transform(X_all)

    # standization, mean = 0, variance = 1
    X_all = scale(X_all)

     # split trainning and test data
    X_train = X_all[0:n_train,:]
    X_test  = X_all[n_train:len(X_all),:]
    print 'X_train.shapre', X_train.shape
    print 'X_test.shapre', X_test.shape
    
    # put revenue back
    Xy_train = np.c_[X_train, y_train]

    imputed_headers = ['Days_Opened', 'City_Group'] + ['P' + str(i) for i in range(1, 38)]

    write_csv('train_i.csv', Xy_train, imputed_headers + ['revenue'], False)
    write_csv('test_i.csv', X_test, imputed_headers, False)