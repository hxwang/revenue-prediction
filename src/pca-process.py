import csv
import sys
import math
import numpy as np 
from sklearn import svm
import preprocess as process
from sklearn import decomposition

rows_train = []
rows_test = []
rows = []



'''
pca reduction of the dataset
'''
def pca_transform(rows)
	global rows
	global rows_train
	global rows_test

	rows_train = process.read_csv_header(train)
	rows_test = process.read_csv_header(test)

	pca = decomposition.PCA()
	
	train_reduced = pca.fit_transform(rows_train)
	test_reduced = pca.transform(rows_test)





