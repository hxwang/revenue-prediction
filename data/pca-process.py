import csv
import sys
import math
import numpy as np 
from sklearn import svm
import preproccess as process
from sklearn import decomposition

rows_train = []
rows_test = []
rows = []



'''
pca reduction of the dataset
'''
def pca_transform(train, test, pca_train, pca_test):
	global rows
	global rows_train
	global rows_test

	# read data
	rows_train = process.read_csv_all(train)
	rows_test = process.read_csv_all(test)


	# do pca 
	pca = decomposition.PCA()
	
	train_reduced = pca.fit_transform(rows_train)
	test_reduced = pca.transform(rows_test)

	# write data
	pca.write_csv(pca_train, train_reduced)
	pca.write_csv(pca_test, test_reduced)




if __name__ == "__main__":
	train = 'train.csv'
	test = 'test.csv'
	pca_train = 'train_pca.csv'
	pca_test = 'test_pca.csv'
	pca_transform(train, test, pca_train, pca_test)



