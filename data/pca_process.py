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
def pca_transform_together(train, test, pca_train, pca_test, cols):

	n_components = 30

	

	# read data
	rows_train = process.read_csv(train)
	rows_test = process.read_csv(test)

	rows_train = np.array(rows_train, dtype = float)
	rows_test = np.array(rows_test,  dtype = float)	

	# seperate revenue
	revenue = rows_train[:,len(rows_train[0])-1]
	rows_train = rows_train[:, cols]	
	print 'rows_train.shape', rows_train.shape
	print 'revenue.shape', revenue.shape
	
	rows_test = rows_test[:, cols]
	print 'rows_test.shape', rows_test.shape

	rows_all = np.vstack((rows_train, rows_test))
	# do pca 
	pca = decomposition.PCA(n_components=n_components)	
	
	all_reduced = pca.fit_transform(rows_all)
	print 'variance = ', pca.explained_variance_ratio_
	print 'total variance = ', np.sum(pca.explained_variance_ratio_)

	train_reduced = rows_all[0:len(rows_train),:]
	test_reduced = rows_all[len(rows_train):len(rows_all),:]
	# put revenue back
	train_reduced = np.c_[train_reduced, revenue]
	# print 'train_reduced.shape', train_reduced.shape

	
	

	# create csv header
	header = ['attr' + str(i) for i in range(n_components)]

	# write data
	process.write_csv(pca_train, train_reduced.tolist(), header + ['revenue'])
	process.write_csv(pca_test, test_reduced.tolist(), header)

def pca_transform_seperate(train, test, pca_train, pca_test, cols):

	n_components = 20

	skip_cols = 0

	# read data
	rows_train = process.read_csv(train)
	rows_test = process.read_csv(test)

	rows_train = np.array(rows_train, dtype = float)
	rows_test = np.array(rows_test,  dtype = float)	

	# seperate revenue
	revenue = rows_train[:,len(rows_train[0])-1]
	rows_train = rows_train[:, cols]	
	print 'rows_train.shape', rows_train.shape
	print 'revenue.shape', revenue.shape
	
	rows_test = rows_test[:, cols]
	print 'rows_test.shape', rows_test.shape

	# do pca 
	pca = decomposition.PCA(n_components=n_components)	
	
	train_reduced = pca.fit_transform(rows_train)
	print 'variance = ', pca.explained_variance_ratio_
	print 'total variance = ', np.sum(pca.explained_variance_ratio_)

	# put revenue back
	train_reduced = np.c_[train_reduced, revenue]
	# print 'train_reduced.shape', train_reduced.shape

	test_reduced = pca.transform(rows_test)
	# print 'test_reduced.shape', test_reduced.shape

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

	cols = {i for i in range(0, 41)}
	open_date = {0}
	city_name = {1}
	city_group = {2}
	city_type = {3}

	print cols

	# cols = cols - open_date
	cols = cols - city_name
	cols = cols - city_type

	print cols

	cols = list(cols)


	
	if(sys.argv[1]=='-t'):
		pca_transform_together(train, test, pca_train, pca_test, cols)
	elif(sys.argv[1]=='-s'):
		pca_transform_seperate(train, test, pca_train, pca_test, cols)
	else:
		print "please input pca type, [-t] is together, [-s] is seperate"


