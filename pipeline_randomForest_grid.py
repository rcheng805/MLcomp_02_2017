from sklearn.preprocessing import OneHotEncoder, normalize, Imputer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
import GridCV_scores_printer
import pandas as pd
import numpy as np
import csv
from scipy import sparse

#Helper function to process data
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

#Helper function to process data
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])

#Load in all the data
def preprocessing_data(rerun=False):
    if rerun:

        ver = pd.read_csv("train_edited_2008.csv", delimiter=',')
        test_data_2008 = pd.read_csv("test_edited_2008.csv", delimiter=',')
        test_data_2012 = pd.read_csv("test_edited_2012.csv", delimiter=',')

        train_shape = ver.shape
        test_data_2008_shape = test_data_2008.shape
        #Separate out the label column, ordered columns, and categorical columns
        label_column = ["PES1"]

        ordered_columns = ["HUFAMINC", "HWHHWGT", "HRNUMHOU", "HUPRSCNT", "GTCBSASZ",
                           "PEAGE", "PEEDUCA", "PRINUSYR", "PEMJNUM", "PEHRUSL1",
                           "PEHRUSL2", "PUHROFF2", "PUHROT2", "PEHRACT1", "PEHRACT2",
                           "PEHRACTT", "PELAYDUR", "PELKDUR", "PRHRUSL",
                           "PRUNEDUR", "PEERNH2", "PEERNH1O", "PRERNHLY", "PWCMPWGT",
                           "PRNMCHLD", "PWVETWGT", "PWSSWGT", "PWORWGT", "PWLGWGT",
                           "PWFMWGT", "PEERNWKP", "PEERNHRO", "HRMIS"]

        train_data_y = np.reshape(ver[label_column].values, [ver.shape[0], ])

        joined_data = ver.drop(label_column, 1).append(test_data_2008).append(test_data_2012)
        
        #Impute blank values for the ordered columns (either setting them to mean or max of remaining data)
        imp_mean = Imputer(missing_values=-1, strategy='mean', axis=0)
        imp_max = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

        negative_conversion_mean = ["HUFAMINC","PUHROFF2","PUHROT2","PEERNH2","PEERNH1O","PRERNHLY"]
        negative_conversion_max = ["PEHRUSL1","PEHRUSL2","PEHRACT1","PEHRACTT","PRHRUSL"]
        one_conversion = ["PEMJNUM"]
        zero_conversion = ["PRINUSYR","PELAYDUR","PELAYDUR","PELKDUR","PRUNEDUR","PRNMCHLD"]

        joined_data[negative_conversion_mean] = joined_data[negative_conversion_mean].replace([-1, -2, -3, -4], [-1, -1, -1, -1])
        joined_data[negative_conversion_max] = joined_data[negative_conversion_max].replace([-1, -2, -3, -4], [-1, -1, -1, -1])
        joined_data[zero_conversion] = joined_data[zero_conversion].replace(-1, 0)
        joined_data[one_conversion] = joined_data[one_conversion].replace(-1, 1)

        joined_data[negative_conversion_mean] = imp_mean.fit_transform(joined_data[negative_conversion_mean])
        joined_data[negative_conversion_max] = imp_max.fit_transform(joined_data[negative_conversion_max])

        #Do one hot encoding of the categorical data
        enc = OneHotEncoder()
        data_ordered = joined_data[ordered_columns]
        data_ordered = normalize(data_ordered.values, axis=0)

        data1 = joined_data.drop(ordered_columns, 1)
        data_categorical = enc.fit_transform(data1.values.astype(np.int64) + 100)

        new_data_x = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(data_ordered), data_categorical]))

        train_data_x = new_data_x[:train_shape[0], :]
        test_data_x_2008 = new_data_x[train_shape[0]:train_shape[0]+test_data_2008_shape[0], :]
        test_data_x_2012 = new_data_x[train_shape[0]+test_data_2008_shape[0]:, :]
        
        #Save the processed data
        save_sparse_csr('train_data_x.npz', train_data_x)
        np.save('train_data_y.npy', train_data_y)
        save_sparse_csr('test_data_x_2008.npz', test_data_x_2008)
        save_sparse_csr('test_data_x_2012.npz', test_data_x_2012)

    else:
        #If we have previously saved the processed data, simply read it in
        train_data_x = load_sparse_csr('train_data_x.npz')
        train_data_y = np.load('train_data_y.npy')
        test_data_x_2008 = load_sparse_csr('test_data_x_2008.npz')
        test_data_x_2012 = load_sparse_csr('test_data_x_2012.npz')

    return train_data_x, train_data_y, test_data_x_2008, test_data_x_2012

#Load Data
train_data_x, train_data_y, test_data_x_2008, test_data_x_2012 = preprocessing_data(rerun=False)

#Set grid parameters to search through
ESTIMATOR_NUM = [10,20,30,40,50,60,70,80,90,100,110,120,130]
LEAF_NUM = [2,4,6,8,10,14,18,22,26,30]
C_VAL = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
param_grid = {
    'feature_selection__estimator__C': C_VAL,
    'classification__n_estimators': ESTIMATOR_NUM,
    'classification__min_samples_leaf': LEAF_NUM
}

#Create pipeline for feature selection and classification
clf = Pipeline([('feature_selection', SelectFromModel(svm.LinearSVC(loss="l2",penalty="l1",dual=False))),
                ('classification', RandomForestClassifier())])
grid = GridSearchCV(clf, param_grid=param_grid,verbose=10,n_jobs=3,cv=4)

#Fit the data for all grid parameters
grid.fit(train_data_x, train_data_y)
score = grid.score(train_data_x, train_data_y)


#Save the results
mean_test = grid.cv_results_['mean_test_score']
mean_train = grid.cv_results_['mean_train_score']
mean_test = np.reshape(mean_test,(mean_test.shape[0],1))
mean_train = np.reshape(mean_train,(mean_train.shape[0],1))

params = grid.cv_results_['params']
params = np.asarray(params)
array = np.asarray(params[0].values())
array = np.reshape(array,(1,array.shape[0]))

for i in range(len(params)-1):
    next_arr = np.asarray(params[i+1].values())
    next_arr = np.reshape(next_arr, (1,next_arr.shape[0]))
    array = np.append(array,next_arr,axis=0)

array = np.append(array, mean_train, axis=1)
array = np.append(array, mean_test, axis=1)
print(array)
np.savetxt('crossVal_Pipeline_Results.csv', array,delimiter=',')
GridCV_scores_printer.print_GridCV_scores(grid,"pipeline_results.csv")

#Print out some data for submission if needed

#test_shape = test_data_x_2008.shape
#np.savetxt('pipeline_forest_test.csv', np.hstack([np.reshape(np.array(range(test_shape[0])), [test_shape[0], 1]),
#                                  np.reshape(clf.predict(test_data_x_2008).astype(np.int64), [test_shape[0], 1])]),
#           fmt='%d', delimiter=',', header='id,PES1', comments='')
