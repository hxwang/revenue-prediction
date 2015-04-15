#!/usr/bin/env python

import csv
import sys
import time
import datetime
from sklearn import preprocessing
import numpy as np


city_names = {}
city_groups = {}
types = {}
rows_train = []
rows_test = []
rows = []


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
return: dictionary
E.g.,
{'a', 'b', 'c', 'a', 'b', 'd'} => {'a':0, 'b':1, 'c':2, 'd':3}
'''
def make_dict(rows, columnId):
    d = {}
    for row in rows:
        k = row[columnId]
        if k in d: continue
        d[k] = len(d)

    return d

def scan_city_name(rows):
    return make_dict(rows, 2)

def scan_city_group(rows):
    return make_dict(rows, 3)

def scan_type(rows):
    return make_dict(rows, 4)


'''
transform nominal value to integer value
'''
def transform_data(rows):

    global city_names
    global city_groups
    global types    

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

    print 'total citys = %s ' % len(city_names)    
    print 'city group = %s ' % city_groups
    print 'types = %s ' % types


    # tranfrom data to numeric types
    rows_train = transform_data(rows_train)
    rows_test = transform_data(rows_test)

    rows_train = np.array(rows_train, dtype = float)
    rows_test = np.array(rows_test, dtype = float)

    X_train = rows_train[:, 0:len(rows_train[0])-1]
    y_train = rows_train[:, len(rows_train[0])-1]
    # normalize data
    X_train = preprocessing.normalize(X_train, norm='l1', axis=0)
    rows_train = np.c_[X_train, y_train]
    
    rows_test = preprocessing.normalize(rows_test, norm='l1', axis=0)


    train_header = read_csv_header(train)
    test_header = read_csv_header(test)
    train_header = train_header[1:len(train_header)]
    test_header = test_header[1:len(test_header)]

    write_csv('train_cleaned.csv', rows_train, train_header, True)
    write_csv('test_cleaned.csv', rows_test, test_header, True)

