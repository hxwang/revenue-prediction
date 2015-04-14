#!/usr/bin/env python

import csv
import sys
import time
import datetime

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

    global rows
    global rows_train
    global rows_test

    rows_train = read_csv(train)
    rows_test = read_csv(test)
    rows = rows_train + rows_test

    return rows

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

    rows = get_all_rows(train, test)
    city_names = scan_city_name(rows)
    city_groups = scan_city_group(rows)
    types = scan_type(rows)

    print 'total citys = %s ' % len(city_names)    
    print 'city group = %s ' % city_groups
    print 'types = %s ' % types

    th = read_csv_header(train)
    th = th[1:len(th)]
    t = transform_data(rows_train)
    write_csv('train_cleaned.csv', t, th, True)

    th = read_csv_header(test)
    th = th[1:len(th)]
    t = transform_data(rows_test)
    write_csv('test_cleaned.csv', t, th, True)

