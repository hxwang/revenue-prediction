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
rows: n*m array
columnId: int
return: use mean revenue to represent the numeric value
E.g.,
{'a', 'b', 'c', 'a', 'b', 'd'} => {'a':0, 'b':1, 'c':2, 'd':3}
'''
def make_dict(rows, columnId, revenueId=42):
    d = {}
    count = {}
    # scan each data
    for row in rows:
        k = row[columnId]
        # if k in d: continue

        if k not in d: 
            d[k] = 0
            count[k] = 0

        if len(row) > revenueId:
            d[k] = d[k] + float(row[revenueId])
            count[k] = count[k] + 1


    # calculate mean_revenue and assign to each numeric value
    revenue_mean = sum(d.values())/sum(count.values())

    print 'revenue_mean = %s d.values = %s count = %s' % (revenue_mean, sum(d.values()), sum(count.values()))

    for k in d:
        if d[k] == 0:
            d[k] = revenue_mean
        else:
            d[k] = d[k]/count[k]

    print d
    return d

'''
rows: n*m array
columnId: int
return: use mean revenue to represent the numeric value
E.g.,
{'a', 'b', 'c', 'a', 'b', 'd'} => {'a':0, 'b':1, 'c':2, 'd':3}
'''
def make_ids(rows, columnId):
    d = {}
    
    # scan each data
    for row in rows:
        k = row[columnId]        
        if k not in d:
            d[k] = len(d)
            
    return d

def scan_city_name(rows):
    return make_ids(rows, 2)

def scan_city_group(rows):
    return make_ids(rows, 3)

def scan_type(rows):
    return make_ids(rows, 4)


'''
transform nominal value to integer value
'''
def transform_data(rows, city_names, city_groups, types):

    for row in rows:        
        row[1] = datetime.datetime.strptime(row[1], '%m/%d/%Y').date()
        row[1] = int(time.mktime(row[1].timetuple()))    
        # transform using dictionary city_names, city_groups, types
        row[2] = city_names[row[2]]
        row[3] = city_groups[row[3]]
        row[4] = types[row[4]] 
    return rows



if __name__ == '__main__':
    train = 'train.csv'
    test = 'test.csv'

    # read data
    rows, rows_train, rows_test = get_all_rows(train, test)
    city_names = scan_city_name(rows)
    city_groups = scan_city_group(rows)
    types = scan_type(rows)

    print 'citys = %s ' % city_names
    print 'city group = %s ' % city_groups
    print 'types = %s ' % types

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

    write_csv('train_raw.csv', rows_train, train_header, True)
    write_csv('test_raw.csv', rows_test, test_header, True)

    # merge train and test in order to do processing together
    X_train = rows_train[:, 0:len(rows_train[0])-1]
    y_train = rows_train[:, len(rows_train[0])-1]
    X_test  = rows_test

    X_all   = np.vstack((X_train, rows_test))
    # print 'X_all.shapre', X_all.shape
    

    # standization, mean = 0, variance = 1
    X_all = scale(X_all)

    # split trainning and test data
    X_train = X_all[0:len(X_train),:]
    X_test  = X_all[len(X_train):len(X_all),:]
    # print 'X_test.shapre', X_test.shape
    
    # put revenue back
    rows_train = np.c_[X_train, y_train]

    

    write_csv('train_scaled.csv', rows_train, train_header, True)
    write_csv('test_scaled.csv', X_test, test_header, True)

