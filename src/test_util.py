import csv
import sys
import math
import numpy as np
import smote
import random
from sklearn import svm
from sklearn import cross_validation
from sklearn import kernel_ridge
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import *
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import *


def test_function():
	a = np.array([0 ,1 ,1, 0])
	b = np.array([2, 3, 4, 5])	
	c = b[a==1]
	d = b[a==0]
	e = np.zeros(4)
	e[a==1] = c*10	
	e[a==0] = d
	print c
	print d
	print e

if __name__ == '__main__':
	test_function()