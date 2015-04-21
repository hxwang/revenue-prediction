#!/usr/bin/env python
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
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.cluster import *


'''
read file, e.g., training file, testing file
return: n*m matrix
'''
def read_csv(filename):    
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()
        for row in reader:
            rows.append(row)
    return rows

'''
write solution file
format: id, predicted value
'''
def write_solution(filename, y):
    print 'writeing solution to %s' % filename
    with open(filename, 'w') as f:
        f.write('Id,Prediction\n')
        for i in range(len(y)):
            f.write(str(i)+','+str(y[i])+'\n')
    print 'done'


def k_folds_two_class(X,y, split_point, k, config):
    labels = binarize(y, split_point)[0]
    n = len(X)
   
    shuffle = False
    if 'shuffle' in config:
        shuffle = True
        
    total_score_arr = []
    # k fold
    kf = cross_validation.KFold(n=n, n_folds=k, shuffle=shuffle)

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = svm.SVC()        
        #model = KNeighborsClassifier(n_neighbors = 5, weights='distance')
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        if 'figure' in config:
            visualize_predict(y_predict, y_test) 

        score = math.sqrt(mean_squared_error(y_predict, y_test))
        print 'score: ', score
        # total_score += score
        total_score_arr.append(score)

    score = np.mean(total_score_arr)
    variance = np.var(total_score_arr)

    print 'avg score =', score , 'var' , variance    

    return total_score_arr

'''
train classification model 
return: trained model
'''
def train_classcification_model(X, y, split_point, config={}):
    labels = binarize(y, split_point)[0]
    print X.shape
    print labels.shape

    # train with all data 
    model = svm.SVC()
    model.fit(X, labels)
    return model       

'''
train the regression model for class0 and class1
return: models for class0 and class1
'''
def train_regress_model(X_train,  y_train, split_point, config):
    y_label = binarize(y_train, split_point)[0]
    X_train_zero = X_train[y_label==0]
    X_train_one = X_train[y_label == 1]
    y_train_zero = y_train[y_label==0]
    y_train_one = y_train[y_label==1]

    
    if 'ensemble' in config:
        model_zero = [
            KNeighborsRegressor(n_neighbors=22, weights='distance'),            
            svm.NuSVR(nu=0.6, C=1e2, degree=2, gamma=0.16),
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=1, random_state=0, loss='lad')
        ]
    else:
        model_zero = svm.NuSVR(nu=0.33, C=9e6, degree=2, gamma=0.0092)
    
    for model in model_zero:
        model.fit(X_train_zero, y_train_zero)
    model_one = svm.NuSVR(nu=0.33, C=9e6, degree=2, gamma=0.0092)
    # model_zero.fit(X_train_zero, y_train_zero)
    model_one.fit(X_train_one, y_train_one)

    return model_zero, model_one

'''
predict the results of test data using two trainied regression model
return: predict result
'''
def predict_test_two_class(X_test, model_zero, model_one, model):
    y_test_label = model.predict(X_test)
    print y_test_label

    X_test_zero = X_test[y_test_label==0]
    X_test_one = X_test[y_test_label==1]

    print 'X_test_zero = %s X_test_one = %s' % (X_test_zero.shape, X_test_one.shape)

    y_predict_zero_all = []
    for m in model_zero:
        y_predict_zero_all.append(m.predict(X_test_zero))
    y_predict_zero = np.mean(np.array(y_predict_zero_all), axis=0)


    # y_predict_zero = model_zero.predict(X_test_zero)
    y_predict_one = model_one.predict(X_test_one)

    y_predict = np.zeros(len(X_test))

    y_predict[y_test_label==0] = y_predict_zero
    y_predict[y_test_label==1] = y_predict_one
    return y_predict

# scale X for knn based on feature importance
def scale_X_knn(X, config):
    if 'feature_importances' not in config: return X

    fi = config['feature_importances']

    TX = np.copy(X)
    for i in range(0, len(X[0])):
        TX[:, i] = TX[:, i] * fi[i]

    return TX



'''
produce model 
X: record, n*m matrix
Y: revenue, n*1 matrix
return: models
'''
def fit(X, y, config = {}):

    models = []

    if 'ensemble' in config:
        
        models = [
            #KNeighborsRegressor(n_neighbors=23, weights='distance'),
            # KNeighborsRegressor(n_neighbors=22, weights='distance'),
            #KNeighborsRegressor(n_neighbors=20, weights = 'distance'),
            #KNeighborsRegressor(n_neighbors=15, weights = 'distance'),   
            
            ########################################################################## 
            # for standardized data
            # SGDClassifier(loss="huber")

            ###########################################################################
            
            # KNeighborsRegressor(n_neighbors=22, weights = 'distance'),
            # svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047),

            # svm.NuSVR(nu=0.33, C=9e6, degree=2, gamma=0.0092),
            # GradientBoostingRegressor(n_estimators=100, learning_rate=0.15, max_depth=1, random_state=1, loss='lad')

            ########################################################################## 
            # for standardized pcaed data
            # KNeighborsRegressor(n_neighbors=22, weights = 'distance'),
            # svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047),
            # GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1, loss='lad')
            # GradientBoostingRegressor(n_estimators=200, learning_rate=0.62, max_depth=1, random_state=1, loss='lad'),
            # RandomForestRegressor(n_estimators=70, n_jobs=-1)
            ###########################################################################

            ########################################################################## 
            # for normalized data
            # KNeighborsRegressor(n_neighbors=21, weights = 'distance'),
            # svm.NuSVR(nu=0.35, C=5e7, degree=2, gamma=0.008),
            # GradientBoostingRegressor(n_estimators=150, learning_rate=0.8, max_depth=1, random_state=1, loss='lad')
            ###########################################################################

            # RandomForestRegressor(n_estimators=100, n_jobs=-1)
            # AdaBoostRegressor(n_estimators=100,  learning_rate = 0.3, loss='exponential')
            # GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=1, loss='lad'),
            # GradientBoostingRegressor(n_estimators=150, learning_rate=0.2, max_depth=3, random_state=1, loss='lad')
            # svm.SVR(C=1.3, degree=3, gamma=0.05)
        ]

        if 'huahua' in config:
            models = [
                KNeighborsRegressor(n_neighbors=22, weights='distance'),
                svm.NuSVR(nu=0.25, C=1.2e7, degree=2, gamma=0.0042),
                GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0, loss='lad'),
            ]

            # best score for pca
            # models = [
            #     KNeighborsRegressor(n_neighbors=30, weights='distance'),            
            #     svm.NuSVR(nu=0.5, C=5e6, degree=2, gamma=0.55),
            #     GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='lad'),
            # ]
            # best record without pca
            # models = [
            #     KNeighborsRegressor(n_neighbors=22, weights='distance'),            
            #     svm.NuSVR(nu=0.6, C=1e2, degree=2, gamma=0.16),
            #     GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=1, random_state=0, loss='lad'),
            # ]
    else:

        model = KNeighborsRegressor(n_neighbors=22, weights='distance')

        # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='lad')
        #model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.6, max_depth=1, random_state=0, loss='lad')

        # model =  RandomForestRegressor(n_estimators=70, n_jobs=-1)

        #model = KNeighborsRegressor(n_neighbors=22, weights = 'distance')
        # model = svm.NuSVR(nu=0.25, C=2.3e7, degree=2, gamma=0.0047)
        # #svm.NuSVR(nu=0.33, C=9e6, degree=2, gamma=0.0092),
        # model = GradientBoostingRegressor(n_estimators=80, learning_rate=0.2, max_depth=1, random_state=1, loss='lad')

        # # avg:  2.36456772543 total_var: 21.5326576549
        # model = BaggingRegressor(n_estimators=100, n_jobs=-1, max_features = 15)
        #model = BaggingRegressor(n_estimators=100, n_jobs=-1, max_features = 15)

        # model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        # model = svm.NuSVR(nu=0.27, C=1.6e7, degree=2, gamma=0.0076)
        # model = svm.NuSVR(nu=0.6, C=1e2, degree=2, gamma=0.16)
        # model = svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047)

        #model = RadiusNeighborsRegressor(radius=15, weights='distance')

        # model = AdaBoostRegressor(n_estimators=500,  learning_rate = 0.3, loss='exponential')

        # model = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0, loss='lad') 

        # score: 4.975
        # model = kernel_ridge.KernelRidge(alpha=50, kernel='linear')

        
        # model = svm.NuSVR(nu=0.27, C=2.3e7, degree=2, gamma=0.0047)

        # score: 3.19637548788
        # model = SGDClassifier(loss="hinge", penalty="l2")

        # score: 2.43654929108 weights = uniform
        # score: 2.37613424202 n_neighbors=20, weights = distance, test: 1730840.19422
        # model = KNeighborsRegressor(n_neighbors=20, weights = 'distance')

        # score: 3.40178270362
        # model = DecisionTreeRegressor()

        # score: 2.54716252715
        #model = svm.SVR(C=1e7, degree=3, gamma=0.0000025)
        # total avg:  2.55070552804 total_var: 37.7768440307

        # Grid Search
        # scorer = make_scorer(mean_squared_error, greater_is_better=False)
        # grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)

        # model = svm.SVR(kernel='poly', coef0 = 60, C=6, degree=2, gamma=0.004)

        models = [ model ]

    # oversampling the training set
    if 'oversampling' in config:
        X, y = smote.SMOTE(X, y, 200, 20)

    if 'logy' in config:
        y = np.log(y)



    for model in models:
        print 'fitting model %s' % type(model).__name__

        TX = X
        if type(model).__name__ == 'KNeighborsRegressor':
            TX = scale_X_knn(X, config)
            
        model.fit(TX, y)

    return models

'''
predict by given models
models: given models
X: record to predict, n*m matrix
prob: whether predict probability
return: predicted (mean) revenue, n*1 matrix
'''
def predict(models, X, config={}, prob=False):

    ys = []

    for model in models:
        print 'predicting...'

        TX = X
        if type(model).__name__ == 'KNeighborsRegressor':
            TX = scale_X_knn(X, config)

        pred = model.predict(TX) if prob == False else model.predict_proba(TX)
        ys.append(pred)

    # return mean prediction
    y = np.mean(np.array(ys), axis=0)

    if 'logy' in config:
        y = np.exp(y)

    return y


'''
k fold validation
return: average Root Mean Squared Eroor (RMSE)
'''
def k_folds_one_class(X, y, k = 5, config={}):
    n = len(X)

    shuffle = True if 'shuffle' in config else False

    kf = cross_validation.KFold(n=n, n_folds=k, shuffle=shuffle)

    # total_score = 0.0
    total_score_arr = []

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        models = fit(X_train, y_train, config)

        y_predict = predict(models, X_test, config)

        if 'figure' in config:
            # visualize predict results
            visualize_predict(y_predict, y_test)
       
        score = math.sqrt(mean_squared_error(y_predict, y_test))
        print 'score: ', score/ 1e6
        # total_score += score
        total_score_arr.append(score)
    score = np.mean(total_score_arr)
    variance = np.var(total_score_arr)

    print 'avg score =', score / 1e6 , 'var' , variance /1e10     


   

    return total_score_arr

'''
visualize predict results
'''
def visualize_predict(y_predict, y_test):
    X = [i for i in range(0, len(y_predict))]
    print 'X', X

    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    plt.scatter(X, y_test, c='k', label='data')
    plt.plot(X, y_predict, c='g', label='prediction')
    plt.axis('tight')
    plt.legend(loc='center left')
    plt.title("Predict v.s. Real")
    plt.show()

def predict_test(X_train, y_train, X_test, config):
    
    # rebuild the model

    models = fit(X_train, y_train, config)
    y_predict = predict(models, X_test, config)

    return y_predict

# resample towards high revenue
def resample(X_train, y_train,  threshold = 9e6, ratio = 0.2):

    X = []
    y = []

    # prob = normalize(np.absolute(scale(y_train)))[0]
    # print prob

    while len(X) < len(X_train) * 2:
        for i in range(0, len(y_train)):
            p = random.random()
            if y_train[i]<8e6 and p > ratio:
                continue
            else:
                X.append(X_train[i])
                y.append(y_train[i])

    return np.array(X), np.array(y)

def parse_arg(argv):
    config = {}
    config['date'] = True
    config['city_group'] = True
    config['city_name'] = True
    config['type'] = True    
    config['kfolds'] = 5

    for i in range(0, len(argv)):
        arg = argv[i]
        if arg == '-t':
            config['test'] = True
        if arg == '-pca':
            config['pca'] = True
        if arg == '-os':
            config['oversampling'] = True
        if arg == '--no-date':
            config.pop('date', None)
        if arg == '--no-city-group':
            config.pop('city_group', None)
        if arg == '-e':
            config['ensemble'] = True
        if arg == '-s':
            config['shuffle'] = True
        if arg == '-r':
            config['repeat'] = True
            config['shuffle'] = True
        if arg == '--no-city-name':
            config.pop('city_name', None)
        if arg == '--no-type':
            config.pop('type', None)
        if arg == '--no-city-info':
            config.pop('type', None)
            config.pop('city_name', None)
            config.pop('city_group', None)
        if arg == '-huahua':
            config['huahua'] = True
        if arg == '-kfolds':
            config['kfolds'] = int(argv[i+1])
        if arg == '-logy':
            config['logy'] = True
        if arg == '-figure':
            config['figure'] = True
        if arg == '-resample':
            config['resample'] = True
        if arg == '-one':
            config['one'] = True
        if arg == '-three':
            config['three'] = True
    return config


def predict_with_one_class(config, X_train, y_train, X_test):
    # calcualte average Root Mean Squared Eror (RMSE)
    total_avg = 0.0

    repeat = 30 if 'repeat' in config else 1
    total_score_arr = []
    for i in range(repeat):
        print '----- repeat %s ----' % i
        score_arr = k_folds_one_class(X_train, y_train, k=config['kfolds'], config=config)
        total_score_arr +=  score_arr
    
    total_score_arr = np.array(total_score_arr)

    print 'total_score_arr shape', total_score_arr.shape
        # total_avg += score_arr
    total_avg = np.mean(total_score_arr)/ 1e6
    total_var = np.var(total_score_arr)/ 1e10
    # total_avg /= repeat

    print "total avg: ", total_avg, "total_var:", total_var

    ################################################
    # run test

    if 'test' in config:
       
        print 'X_test.shape', X_test.shape
        y_predict = predict_test(X_train, y_train, X_test, config)
        return y_predict


def predict_with_two_class(config, X_train, y_train, X_test, class_split_threshold):
    # calcualte average Root Mean Squared Eror (RMSE)
    total_avg = 0.0     

    repeat = 30 if 'repeat' in config else 1
    total_score_arr = []
    for i in range(repeat):
        print '----- repeat %s ----' % i
        score_arr = k_folds_two_class(X_train, y_train, class_split_threshold,  k=config['kfolds'], config=config)
        total_score_arr +=  score_arr
    
    total_score_arr = np.array(total_score_arr)

    print 'total_score_arr shape', total_score_arr.shape
        # total_avg += score_arr
    total_avg = np.mean(total_score_arr)
    total_var = np.var(total_score_arr)
    # total_avg /= repeat

    print "total avg: ", total_avg, "total_var:", total_var

    ################################################
    # run test

    if 'test' in config:
        print 'X_test.shape', X_test.shape
        classify_model = train_classcification_model(X_train, y_train, class_split_threshold, config)
        model_zero, model_one = train_regress_model(X_train, y_train, class_split_threshold, config)        
        y_predict = predict_test_two_class(X_test, model_zero, model_one, classify_model)
        return y_predict


def kfolds(models, X, y, config={}, measure = 'RMSE'):
    repeat = 30 if 'repeat' in config else 1
    total_score_arr = []
    for i in range(repeat):
        print '----- repeat %s ----' % i
        
        n = len(X)
        k = config['kfolds']

        shuffle = True if 'shuffle' in config else False

        kf = cross_validation.KFold(n=n, n_folds=k, shuffle=shuffle)
    
        score_arr = []

        predict_proba = measure == 'PROBA'

        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for m in models: m.fit(X_train, y_train)
            y_predict = predict(models, X_test, config, predict_proba)

            if measure == 'RMSE':
                score = math.sqrt(mean_squared_error(y_predict, y_test))
                print 'score: ', score/ 1e6
            elif measure == 'PROBA':                
                # score = brier_score_loss(y_test, y_predict[:,1])
                y_pred_labels = np.zeros(len(y_test))
                y_pred_labels[y_predict[:, 1]>=0.5] = 1

                # print 'prob = ', y_predict
                # print 'predicted_labels = ', y_pred_labels                
                # print 'actual_labels = ', y_test

                score = np.sum(np.absolute(np.subtract(y_test, y_pred_labels)))

                # use labels as result for visualization
                y_predict = y_pred_labels

                print 'classification error: ', score

            score_arr.append(score)
            

            if 'figure' in config:
                # visualize predict results
                visualize_predict(y_predict, y_test)

        total_score_arr +=  score_arr

    total_avg = np.mean(total_score_arr)/ 1e6
    total_var = np.var(total_score_arr)/ 1e10    

    print "total avg: ", total_avg, "total_var:", total_var

    return total_avg, total_var

def predict_with_three_classes(X_train, y_train, X_test, config):
    # 3 x 1 matrix
    split_points = np.percentile(y_train, [90])

    print 'split_points', split_points

    index1, index2 = y_train <= split_points[0], y_train>split_points[0]

    X_train1, X_train2 = X_train[index1], X_train[index2]
    y_train1, y_train2 = y_train[index1], y_train[index2]

    labels = np.zeros(len(y_train))

    labels[index1], labels[index2] = 0, 1

    models1 = [
        KNeighborsRegressor(n_neighbors=20, weights='distance'),            
        svm.NuSVR(nu=0.2, C=2e6, degree=2, gamma=0.2),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=1, random_state=0, loss='lad'),
    ]

    total_avg1, total_var1 = kfolds(models1, X_train1, y_train1, config)

    print '------------------------------'

    # for high revenue
    models2 = [
        KNeighborsRegressor(n_neighbors=8, weights='distance'),                    
        GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=1, random_state=0, loss='lad'),
        svm.NuSVR(nu=0.7, C=5e8, degree=2, gamma=0.00005)
    ]

    total_avg2, total_var2 = kfolds(models2, X_train2, y_train2, config)
    
    print '------------------------------'

    print 'p_models1 avg/var = %s / %s' % (total_avg1, total_var1)

    print 'p_models2 avg/var = %s / %s' % (total_avg2, total_var2)


    # dbscan = DBSCAN(eps=10)
    # clustering_results = dbscan.fit_predict(X_train)    
    # print 'clustring count "-1" = %s "0" = %s "1" = %s' % (np.sum(clustering_results==-1), np.sum(clustering_results==0), np.sum(clustering_results==1))
    # print 'labels = 0 and cluser = 1 count = %s' % (np.sum(np.subtract(clustering_results, labels)==1))
    # print y_train[clustering_results==1]
    # for i in range(0, len(clustering_results)):    
    #     print labels[i], clustering_results[i]

    c_models = [
        KNeighborsClassifier(n_neighbors=5, weights='distance'),        
        BaggingClassifier(n_estimators=100),
        #GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1)
        ]
    
    total_avg_c, total_var_c = kfolds(c_models, X_train, labels, config = config, measure = 'PROBA')

    print 'c_models avg/var = %s / %s' % (total_avg_c, total_var_c)

    if 'test' not in config: return    

    for m in models1:
        m.fit(X_train1, y_train1)

    for m in models2:
        m.fit(X_train2, y_train2)

    for c_model in c_models:
        c_model.fit(X_train, labels)

    # predict probbility
    y_pred_proba = predict(c_models, X_test, config, prob = True)    

    predict_one = predict(models1, X_test, config)
    predict_two = predict(models2, X_test, config)

    # predict revenue
    y_pred = np.sum(
        [
            np.multiply(predict_one, y_pred_proba[:,0]),
            np.multiply(predict_two, y_pred_proba[:,1])
        ],
        axis = 0)

    # y_pred = np.zeros(len(X_test))
    # low_index, high_index = y_pred_proba[:,1]<0.5, y_pred_proba[:,1]>=0.5

    # print 'high_rev_count = ', np.sum(high_index)

    # y_pred[low_index] = predict_one[low_index]
    # y_pred[high_index] = predict_two[high_index]    

    return y_pred

        

if __name__ == '__main__':

    config = parse_arg(sys.argv)

    train_filename = '../data/train_cleaned.csv'
    test_filename = '../data/test_cleaned.csv'

    if 'pca' in config:
        train_filename = '../data/pca_train.csv'
        test_filename = '../data/pca_test.csv'

    # read training data and convert to numpy array
    train_data = np.array(read_csv(train_filename), dtype=float)
    
   

    if 'pca' in config:
        # if use pca, use all features
        cols = [i for i in range(0, len(train_data[0])-1)]
    else:        

        # selected features
        cols = [] 
        
        if 'date' in config: cols = cols + [0]
        if 'city_name' in config: cols = cols + [1]
        if 'city_group' in config: cols = cols + [2]
        if 'type' in config: cols = cols + [3]

        cols = cols + [i for i in range(4, len(train_data[0])-1)]
    
    print len(cols)
    
    
    # split training data to (record) and (predicted value)
    X_train = train_data[:, cols]
    y_train = train_data[:, len(train_data[0])-1]

    # feature selection
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    feature_importances = np.square(np.array(rfc.feature_importances_))
    config['feature_importances'] = feature_importances
    print feature_importances

    # settings
    class_split_threshold = 17e6

    #resample
    if('resample' in config):
        X_train, y_train = resample(X_train, y_train, threshold = class_split_threshold)

    # train_classcification_model(X_train, y_train, threshold = class_split_threshold,  config = config)
    X_test = []
    if 'test' in config:
        X_test = np.array(read_csv(test_filename), dtype=float)
        X_test = X_test[:, cols]

    print X_train.shape
    if 'one' in config:
        y_predict = predict_with_one_class(config, X_train, y_train, X_test)
    elif 'three' in config:
        y_predict = predict_with_three_classes(X_train, y_train, X_test, config)
    else:
        y_predict = predict_with_two_class(config, X_train, y_train, X_test, class_split_threshold)

    if 'test' in config:
        solution_filename = 'solution.csv'
        print 'y_predict-share', y_predict.shape

        write_solution(solution_filename, y_predict)



