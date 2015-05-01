import csv
import sys
import time
import datetime
from sklearn.decomposition import PCA
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

        # days opened...
        row[1] = (time.mktime((2014,2,1,0,0,0,0,0,0)) - time.mktime(row[1].timetuple())) /  24 / 3600
        # transform using dictionary city_names, city_groups, types
        row[2] = city_names[row[2]]
        row[3] = city_groups[row[3]]
        row[4] = types[row[4]] 
    return rows