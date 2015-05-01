import csv
import sys
import math
import numpy as np 
from sklearn import svm
import preproccess as process
from sklearn import decomposition



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