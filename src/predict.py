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
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.cluster import *
from sklearn.gaussian_process import GaussianProcess
from sklearn import mixture
from sklearn.linear_model import BayesianRidge, LinearRegression


'''
read file, e.g., training file, testing file
return: n*m matrix
'''
def read_csv(filename):
    print 'reading csv from %s' % filename
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


'''
visualize predict results
'''
def visualize_predict(y_predict, y_test):
    import matplotlib.pyplot as plt
    import matplotlib as mtb
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 36}

    mtb.rc('font', **font)

    X = [i for i in range(0, len(y_predict))]
    print 'X', X

    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    plt.scatter(X, y_test, c='k', label='Ground Truth',linewidth=4)
    plt.plot(X, y_predict, c='g', label='Prediction',linewidth=3)
    plt.axis('tight')
    plt.legend(loc='center left')
    plt.title("Prediction v.s. Ground Truth", fontsize=40)
    plt.xlabel('Instances', fontsize=40)
    plt.ylabel('Revenue', fontsize=40)
    plt.show()

'''   
scale X for knn based on feature importance
'''
def scale_X_knn(X, config):
    if 'feature_importances' not in config: return X

    fi = config['feature_importances']

    TX = np.copy(X)
    for i in range(0, len(X[0])):
        TX[:, i] = TX[:, i] * fi[i]

    return TX

# resample towards high revenue
def resample(X_train, y_train,  threshold = 10e6, ratio = 0.8, size = 2):

    X = []
    y = []

    # prob = normalize(np.absolute(scale(y_train)))[0]
    # print prob

    while len(X) < len(X_train) * (size-ratio):
        for i in range(0, len(y_train)):
            p = random.random()
            if y_train[i]<9e6 and p > ratio:
                continue
            else:
                X.append(X_train[i])
                y.append(y_train[i])

    return np.array(X), np.array(y)

'''
fit all the models with given data
'''
def fit_models(models, X, y, config = {}):
    if 'logy' in config:
        y = np.log(y)

    for model in models:
        print 'fitting model %s' % type(model).__name__

        TX = X
        if 'weightfeature' in config:     
            if type(model).__name__ == 'KNeighborsRegressor':
                TX = scale_X_knn(X, config)
                
        model.fit(TX, y)


'''
predict by given models
models: given models
X: record to predict, n*m matrix
prob: whether predict probability
return: predicted (mean) revenue, n*1 matrix
'''
def predict(models, X, config={}, prob=False, weights=None):

    ys = []

    for model in models:
        print 'predicting...'

        TX = X
        if type(model).__name__ == 'KNeighborsRegressor':
            TX = scale_X_knn(X, config)

        pred = model.predict(TX) if prob == False else model.predict_proba(TX)
        ys.append(pred)

    # return mean prediction
    ys = np.array(ys).T
    
    if len(models)==3 and not weights:
        weights = np.matrix([0.2, 0.3, 0.5]).T

    if weights is not None:
        y = np.dot(ys, weights)
        print y.shape
    else:
        y = np.mean(ys.T, axis=0)

    if 'logy' in config:
        y = np.exp(y)

    return np.squeeze(np.asarray(y))


def predict_test(X_train, y_train, X_test, config):
    
    # rebuild the model

    models = fit(X_train, y_train, config)
    y_predict = predict(models, X_test, config)

    return y_predict


'''
kfolds
'''
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

            fit_models(models, X_train, y_train, config)

            y_predict = predict(models, X_test, config, predict_proba)

            if 'logy' in config:
                y_predict = np.exp(y_predict)

            if measure == 'RMSE':
                score = math.sqrt(mean_squared_error(y_predict, y_test))
                print 'score: ', score/ 1e6
            elif measure == 'PROBA':                
                # score = brier_score_loss(y_test, y_predict[:,1])
                y_pred_labels = np.zeros(len(y_test))
                y_pred_labels[y_predict[:, 1]>=0.4] = 1

                # print 'prob = ', y_predict
                # print 'predicted_labels = ', y_pred_labels                
                # print 'actual_labels = ', y_test

                score = np.sum(np.absolute(np.subtract(y_test, y_pred_labels)))

                # use labels as result for visualization
                y_predict = y_pred_labels

                print 'classification error: ', score

            score_arr.append(score)
            
            print y_predict.tolist()
            print y_test.tolist()

            if 'figure' in config:
                # visualize predict results
                visualize_predict(y_predict, y_test)

        total_score_arr +=  score_arr

    total_avg = np.mean(total_score_arr)/ 1e6
    total_var = np.var(total_score_arr)/ 1e10    

    print "total avg: ", total_avg, "total_var:", total_var

    return total_avg, total_var


'''
training and prediction
'''

def predict_with_one_class(config, X_train, y_train, X_test):
    C = np.max(y_train) - np.min(y_train)
    print C

    if 'ensemble' in config:
        models = [        
            KNeighborsRegressor(n_neighbors=22, weights='distance'),
            svm.NuSVR(nu=0.25, C=C, degree=2, gamma=0.01),
            GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss='lad'),

            # KNeighborsRegressor(n_neighbors=22, weights='distance'), svm.NuSVR(nu=0.35, C=C, degree=2, gamma=0.01), #GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=1, random_state=0, loss='lad'), GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5, max_depth=3, random_state=0, loss='lad'),
            #xgb.XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.02, subsample = 0.9, base_score=4.4e6),
            #xgb.XGBRegressor(max_depth=6, learning_rate=0.05)
            # BayesianRidge()
            #GaussianProcess(corr='absolute_exponential')
        ]

        if 'low' in config:
            models = [
                KNeighborsRegressor(n_neighbors=21, weights='distance'),
                svm.NuSVR(nu=0.38, C=C, gamma=0.004),
                GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='lad', subsample=0.95),
            ]    
    else:
        models = [
            # KNeighborsRegressor(n_neighbors=22, weights='distance'),
            # svm.NuSVR(nu=0.38, C=C, gamma=0.004),
            #GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='lad', subsample=0.95),

            # KNeighborsRegressor(n_neighbors=25, weights='distance'),
            # svm.NuSVR(nu=0.35, C=C, gamma=0.01),
            # GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='lad'),

            #GradientBoostingRegressor(n_estimators=50, learning_rate=0.01, max_depth=1, random_state=0, loss='lad'),

            #svm.SVR(C=C, epsilon=0.0001, degree=2, gamma=0.02),
            # RandomForestRegressor(n_estimators=100, min_weight_fraction_leaf=0.1, n_jobs=4)
            GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss='lad'),
            #svm.NuSVR(nu=0.25, C=C, degree=2, gamma=0.01),
            #KNeighborsRegressor(n_neighbors=21, weights='distance'),

            # GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5, max_depth=3, random_state=0, loss='lad'),

        ]

        if 'xgb' in config:            
            import xgboost as xgb
            models = [
                xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.01, subsample = 0.9, min_child_weight=5.0)
            ]

    total_avg, total_var = kfolds(models, X_train, y_train, config)

    print '------------------------------'    

    print 'p_models avg/var = %s / %s' % (total_avg, total_var)
    
    if 'test' not in config: return    

    fit_models(models, X_train, y_train, config)        

    y_pred = predict(models, X_test, config)    

    return y_pred

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
        KNeighborsRegressor(n_neighbors=2, weights='distance'),                    
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='lad'),
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
        BaggingClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.05)
        # svm.SVC(probability=True, class_weight=balance_weights, C = 50, kernel='rbf')
        # RandomForestClassifier(n_estimators = 10, class_weight  = 'subsample')
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
   
def parse_arg(argv):
    config = {}
    config['date'] = True
    config['city_group'] = True
    config['city_name'] = False
    config['type'] = False    
    config['kfolds'] = 5
    config['one'] = True

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
        if arg == '-xgb':
            config['xgb'] = True
        if arg == '-three':
            config['three'] = True
        if arg == '-weightfeature':
            config['weightfeature'] = True
        if arg == '-raw':
            config['raw'] = True
        if arg == '-i':
            config['imputation'] = True
        if arg == '-low':
            config['low'] = True
    return config

def main():
    config = parse_arg(sys.argv)

    train_filename = '../data/train_scaled.csv'
    test_filename = '../data/test_scaled.csv'

    if 'pca' in config:
        train_filename = '../data/train_pca.csv'
        test_filename = '../data/test_pca.csv'

    # if 'raw' in config:
    #     train_filename = '../data/train_raw.csv'
    #     test_filename = '../data/train_raw.csv'    

    if 'imputation' in config:
        train_filename = '../data/train_i.csv'
        test_filename = '../data/test_i.csv'

    # read training data and convert to numpy array
    train_data = np.array(read_csv(train_filename), dtype=float)   

    if 'pca' in config:
        # if use pca, use all features
        cols = [i for i in range(0, len(train_data[0])-1)]
    elif 'imputation' in config:
        # if use imputation, use all features
        cols = [i for i in range(0, len(train_data[0])-1)]
    else:        

        # selected features
        cols = [] 
        
        if 'date' in config: cols = cols + [0]
        if 'city_name' in config: cols = cols + [1]
        if 'city_group' in config: cols = cols + [2]
        if 'type' in config: cols = cols + [3]

        cols = cols + [i for i in range(4, len(train_data[0])-1)]

    
    # split training data to (record) and (predicted value)
    X_train = train_data[:, cols]
    y_train = train_data[:, len(train_data[0])-1]

    # only use low values
    if 'low' in config:
        idx = y_train < 9e6
        X_train, y_train = X_train[idx], y_train[idx]


    # feature selection
    # rfc = RandomForestClassifier()
    # rfc.fit(X_train, y_train)
    # feature_importances = np.square(np.array(rfc.feature_importances_))
    # config['feature_importances'] = feature_importances
    # print feature_importances

    # settings
    class_split_threshold = 17e6

    #resample
    if('resample' in config):
        X_train, y_train = resample(X_train, y_train, threshold = class_split_threshold, size = 2)

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


if __name__ == '__main__':
    main()