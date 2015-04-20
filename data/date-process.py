#!/usr/bin/env python

import csv
import sys
import time
import datetime
from sklearn.preprocessing import *
import numpy as np


def read_csv_all(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    return rows


def read_csv_header(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = reader.next()
        return header

def read_csv(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            rows.append(row)
    return rows

def write_csv(filename, rows, header=None, removeId=False):
    print 'writing to %s' % filename
    start_idx = 1 if removeId else 0
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        if header != None:            
            writer.writerow(header)
        for row in rows:
            writer.writerow(row[start_idx:len(row)])            

def get_all_rows(train, test):

    rows_train = read_csv(train)
    rows_test = read_csv(test)
    rows = rows_train + rows_test

    return rows, rows_train, rows_test




'''
transform nominal value to integer value
'''
def transform_data(rows):

    for row in rows:        
        row[1] = datetime.datetime.strptime(row[1], '%m/%d/%Y').date()
        row[1] = int(time.mktime(row[1].timetuple()))
        # transform using dictionary city_names, city_groups, types
    return rows



if __name__ == '__main__':
    train = 'train.csv'
    test = 'test.csv'

    # read data
    rows, rows_train, rows_test = get_all_rows(train, test)
    


    # tranfrom data to numeric types
    rows_train = np.array(transform_data(rows_train))
    rows_test = np.array(transform_data(rows_test))
    

    # # convert to float type
    # rows_train = np.array(rows_train, dtype = float)
    # rows_test = np.array(rows_test, dtype = float)
    # # print 'rows_test.shapre', rows_test.shape


    train_header = read_csv_header(train)
    test_header = read_csv_header(test)
    train_header = train_header[1:len(train_header)-1]
    test_header = test_header[1:len(test_header)]

    y_train = rows_train[:, len(rows_train[0])-1]
    rows_train = rows_train[:, 0:len(rows_train[0])-1]

    rows_all = np.vstack((rows_train, rows_test)) 

    # write_csv('train_date.csv', rows_train, train_header, True)
    # write_csv('test_date.csv', rows_test, test_header, True)


    # write_csv('all_date_transfered.csv', rows_all, test_header, True)


    pca_file = 'pca_all_date_transfered.csv'
    # pca_file = 'pca_all_date_transfered_no_cityname.csv'
    pca_all = np.array(read_csv(pca_file))
    pca_train = pca_all[0:len(rows_train), :]
    print 'pca_train.shape', pca_train.shape
    print 'y_train.shape', y_train.shape
    pca_train = np.c_[pca_train, y_train]
    print 'pca_train.shape', pca_train.shape
    pca_test = pca_all[len(rows_train):len(pca_all),:]

    pca_header = ['attr'+str(i) for i in range(0, len(pca_all[0]))]
    write_csv('pca_train.csv', pca_train, pca_header+['revenue'], False)
    write_csv('pca_test.csv', pca_test, pca_header, False)



