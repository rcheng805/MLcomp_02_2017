from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import GridCV_scores_printer
from preprocessing_data import preprocessing_data

train_data_x, train_data_y, test_data_x_2008, test_data_x_2012 = preprocessing_data()

rf = RandomForestClassifier()
feat_selection = SelectFromModel(rf)
clf = AdaBoostClassifier(n_estimators=50, learning_rate=0.5)

model = Pipeline([
          ('fs', feat_selection),
          ('clf', clf),
        ])

params = {
    'fs__estimator__n_estimators': range(10, 100, 20),
    'fs__estimator__min_samples_leaf': [2**x for x in range(1, 8)],
    'clf__n_estimators': range(40, 200, 20),
    'clf__learning_rate': [x/10. for x in range(1, 10, 2)]
}

gs = RandomizedSearchCV(model, params, n_iter=100, verbose=10, n_jobs=2)
gs.fit(train_data_x, train_data_y)

result = gs.cv_results_
GridCV_scores_printer.print_GridCV_scores(clf, "results_adaboost_1.csv")
print(result)
