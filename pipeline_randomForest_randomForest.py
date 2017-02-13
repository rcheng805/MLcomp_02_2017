from sklearn.ensemble import RandomForestClassifier
from preprocessing_data import preprocessing_data
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
import GridCV_scores_printer

train_data_x, train_data_y, test_data_x_2008, test_data_x_2012 = preprocessing_data(rerun=False)

rf = RandomForestClassifier()
feat_selection = SelectFromModel(rf)
clf = RandomForestClassifier()

model = Pipeline([
          ('fs', feat_selection),
          ('clf', clf),
        ])

parameters = {
    'fs__estimator__n_estimators': range(10, 100, 20),
    'fs__estimator__min_samples_leaf': [2**x for x in range(1, 10, 2)],
    'clf__n_estimators': range(10, 100, 20),
    'clf__min_samples_leaf': [2**x for x in range(1, 10, 2)]
}

# parameters = {
#     'fs__estimator__n_estimators': [10, 1],
#     'fs__estimator__min_samples_leaf': [2**8],
#     'clf__n_estimators': [10],
#     'clf__min_samples_leaf': [2**10]
# }

grid = GridSearchCV(model, parameters, verbose=10, n_jobs=2)
grid.fit(train_data_x, train_data_y)

score = grid.score(train_data_x, train_data_y)
params = grid.cv_results_['params']
mean_test = grid.cv_results_['mean_test_score']
mean_train = grid.cv_results_['mean_train_score']
split0_test = grid.cv_results_['split0_test_score']
split0_train = grid.cv_results_['split0_train_score']
split1_test = grid.cv_results_['split1_test_score']
split1_train = grid.cv_results_['split1_train_score']
split2_test = grid.cv_results_['split2_test_score']
split2_train = grid.cv_results_['split2_train_score']
mean_test = np.reshape(mean_test, (mean_test.shape[0], 1))
mean_train = np.reshape(mean_train, (mean_train.shape[0], 1))
split0_test = np.reshape(split0_test, (split0_test.shape[0], 1))
split0_train = np.reshape(split0_train, (split0_train.shape[0], 1))
split1_test = np.reshape(split1_test, (split1_test.shape[0], 1))
split1_train = np.reshape(split1_train, (split1_train.shape[0], 1))
split2_test = np.reshape(split2_test, (split2_test.shape[0], 1))
split2_train = np.reshape(split2_train, (split2_train.shape[0], 1))

params = np.asarray(params)
keys = params[0].keys()
with open('people.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(params)

array = np.genfromtxt('people.csv', delimiter=',', skip_header=1)
array = np.append(array, mean_train, axis=1)
array = np.append(array, mean_test, axis=1)
array = np.append(array, split0_train, axis=1)
array = np.append(array, split0_test, axis=1)
array = np.append(array, split1_train, axis=1)
array = np.append(array, split1_test, axis=1)
array = np.append(array, split2_train, axis=1)
array = np.append(array, split2_test, axis=1)

np.savetxt('crossVal_Pipeline_rf_rf_Results', array, fmt='%.8f')
GridCV_scores_printer.print_GridCV_scores(grid, "pipeline_rf_rf_results_2.csv")

print grid.best_params_

test_shape = test_data_x_2008.shape
np.savetxt('pipeline_rf_rf_2008_test.csv', np.hstack([np.reshape(np.array(range(test_shape[0])), [test_shape[0], 1]),
           np.reshape(grid.predict(test_data_x_2008).astype(np.int64), [test_shape[0], 1])]),
           fmt='%d', delimiter=',', header='id,PES1', comments='')

test_shape = test_data_x_2012.shape
np.savetxt('pipeline_rf_rf_2012_test.csv', np.hstack([np.reshape(np.array(range(test_shape[0])), [test_shape[0], 1]),
           np.reshape(grid.predict(test_data_x_2012).astype(np.int64), [test_shape[0], 1])]),
           fmt='%d', delimiter=',', header='id,PES1', comments='')