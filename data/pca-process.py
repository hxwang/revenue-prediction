#!/usr/bin/env python

import csv
import sys
import math
import numpy as np 
from sklearn import svm
import preproccess as process
from sklearn import decomposition

'''
pca reduction of the dataset
'''
def pca_transform(train, test, pca_train, pca_test):

	n_components = 15

	skip_cols = 0

	# read data
	rows_train = process.read_csv(train)
	rows_test = process.read_csv(test)

	rows_train = np.array(rows_train, dtype = float)
	rows_test = np.array(rows_test,  dtype = float)	

	# seperate revenue
	revenue = rows_train[:,len(rows_train[0])-1]
	rows_train = rows_train[:, skip_cols:len(rows_train[0])-1]	
	print 'rows_train.shape', rows_train.shape
	print 'revenue.shape', revenue.shape
	
	rows_test = rows_test[:, skip_cols:len(rows_test[0])]
	print 'rows_test.shape', rows_test.shape

	# do pca 
	pca = decomposition.PCA(n_components=n_components)	
	
	train_reduced = pca.fit_transform(rows_train)
	print 'variance = ', pca.explained_variance_ratio_
	print 'total variance = ', np.sum(pca.explained_variance_ratio_)

	# put revenue back
	train_reduced = np.c_[train_reduced, revenue]
	print 'train_reduced.shape', train_reduced.shape

	test_reduced = pca.transform(rows_test)
	print 'test_reduced.shape', test_reduced.shape

	# create csv header
	header = ['attr' + str(i) for i in range(n_components)]

	# write data
	process.write_csv(pca_train, train_reduced.tolist(), header + ['revenue'])
	process.write_csv(pca_test, test_reduced.tolist(), header)

if __name__ == "__main__":
	train = 'train_cleaned.csv'
	test = 'test_cleaned.csv'
	pca_train = 'train_pca.csv'
	pca_test = 'test_pca.csv'
	pca_transform(train, test, pca_train, pca_test)



