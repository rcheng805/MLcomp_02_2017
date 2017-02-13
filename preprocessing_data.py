from sklearn.preprocessing import OneHotEncoder, normalize, Imputer
import pandas as pd
import numpy as np
from scipy import sparse


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])


def preprocessing_data(rerun=False):
    if rerun:

        ver = pd.read_csv("train_edited_2008.csv", delimiter=',')
        test_data_2008 = pd.read_csv("test_edited_2008.csv", delimiter=',')
        test_data_2012 = pd.read_csv("test_edited_2012.csv", delimiter=',')

        train_shape = ver.shape
        test_data_2008_shape = test_data_2008.shape

        label_column = ["PES1"]

        ordered_columns = ["HUFAMINC", "HRNUMHOU", "HUPRSCNT", "GTCBSASZ",
                           "PEAGE", "PEEDUCA", "PRINUSYR", "PEMJNUM", "PEHRUSL1",
                           "PEHRUSL2", "PUHROFF2", "PUHROT2", "PEHRACT1", "PEHRACT2",
                           "PEHRACTT", "PELAYDUR", "PELKDUR", "PRHRUSL",
                           "PRUNEDUR", "PEERNH2", "PEERNH1O", "PRERNHLY",
                           "PRNMCHLD", "PEERNWKP", "PEERNHRO", "HRMIS"]

        train_data_y = np.reshape(ver[label_column].values, [ver.shape[0], ])

        joined_data = ver.drop(label_column, 1).append(test_data_2008).append(test_data_2012)

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

        enc = OneHotEncoder()
        data_ordered = joined_data[ordered_columns]
        data_ordered = normalize(data_ordered.values, axis=0)

        data1 = joined_data.drop(ordered_columns, 1)
        data_categorical = enc.fit_transform(data1.values.astype(np.int64) + 100)

        new_data_x = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(data_ordered), data_categorical]))

        train_data_x = new_data_x[:train_shape[0], :]
        test_data_x_2008 = new_data_x[train_shape[0]:train_shape[0]+test_data_2008_shape[0], :]
        test_data_x_2012 = new_data_x[train_shape[0]+test_data_2008_shape[0]:, :]

        save_sparse_csr('train_data_x.npz', train_data_x)
        np.save('train_data_y.npy', train_data_y)
        save_sparse_csr('test_data_x_2008.npz', test_data_x_2008)
        save_sparse_csr('test_data_x_2012.npz', test_data_x_2012)

    else:
        train_data_x = load_sparse_csr('train_data_x.npz')
        train_data_y = np.load('train_data_y.npy')
        test_data_x_2008 = load_sparse_csr('test_data_x_2008.npz')
        test_data_x_2012 = load_sparse_csr('test_data_x_2012.npz')

    return train_data_x, train_data_y, test_data_x_2008, test_data_x_2012
