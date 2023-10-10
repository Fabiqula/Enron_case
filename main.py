#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
# from tester import dump_classifier_and_data
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### 1. Select what features you'll use.

features_list = ['poi', 'salary', 'total_payments', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

### 2. Load the dictionary containing the dataset, flatten dictionary and load it into Pandas.

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict

my_dataset_list = []
for key, value in my_dataset.items():
    record = value
    record['person'] = key
    my_dataset_list.append(record)

df = pd.DataFrame(my_dataset_list)

# Moving the person column up front.

temp_col = df['person']
df.drop('person', axis=1, inplace=True)
df.insert(0, 'person', temp_col, allow_duplicates=False)

# Explore the data
df = df.replace('NaN', np.nan)
df.info()

print(sum(df.poi))


### 3. Remove the outliers
features_list = ['salary', 'bonus']
data = featureFormat(my_dataset, features_list, sort_keys = True)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.title('Salary vs Bonus')

outlier = max(data, key=lambda x: x[1])

for person in my_dataset:
    if my_dataset[person]['salary'] != "NaN" and my_dataset[person]['salary'] > 2_500_000:
        print(person)

print(len(my_dataset))
my_dataset.pop('TOTAL', 0)
print(len(my_dataset))

features_list = ['salary', 'bonus']
data = featureFormat(my_dataset, features_list, sort_keys = True)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.title('Salary vs Bonus')
plt.show()

for person in my_dataset:
    if my_dataset[person]['salary'] != 'NaN' and my_dataset[person]['salary'] > 1_000_000:
        print(person)

### 4. Create new feature(s)

for person in my_dataset:
    if my_dataset[person]['from_poi_to_this_person'] != 'NaN' and my_dataset[person]['from_this_person_to_poi'] != 'NaN':
        my_dataset[person]['from_poi_ratio'] = my_dataset[person]['from_poi_to_this_person'] / my_dataset[person]['to_messages']
        my_dataset[person]['to_poi_ratio'] = my_dataset[person]['from_this_person_to_poi'] / my_dataset[person]['from_messages']
    else:
        my_dataset[person]['from_poi_ratio'] = 'NaN'
        my_dataset[person]['to_poi_ratio'] = 'NaN'


### Extract features and labels from dataset for local testing
features_list = ['poi', 'salary', 'total_payments', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'from_poi_ratio', 'to_poi_ratio']
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### 5. Try a varity of classifiers

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=43)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
gnb_pred = clf.predict(features_test)
gnb_score = clf.score(features_test, labels_test)
gnb_precision = precision_score(labels_test, gnb_pred)
gnb_recall = recall_score(labels_test, gnb_pred)
print ('GaussianNB accuracy:', gnb_score)
print ('GaussianNB precision:', gnb_precision)
print ('GaussianNB recall:', gnb_recall, '\n')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
fit = clf.fit(features_train, labels_train)
dt_pred = clf.predict(features_test, labels_test)
dt_score = clf.score(features_test, labels_test)
dt_precision = precision_score(labels_test, dt_pred)
dt_recall = recall_score(labels_test, dt_pred)
print ('Decision Tree accuracy:', dt_score)
print ('Decision Tree precision:', dt_precision)
print ('Decision Tree recall:', dt_recall, '\n')


from sklearn.svm import SVC
clf = SVC(random_state=0)
fit = clf.fit(features_train, labels_train)
svc_pred = clf.predict(features_test)
svc_score = clf.score(features_test, labels_test)
svc_precision = precision_score(labels_test, svc_pred)
svc_recall = recall_score(labels_test, svc_pred)
print ('SVC accuracy:', svc_score)
print ('SVC precision:', svc_precision)
print ('SVC recall:', svc_recall, '\n')


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
fit = clf.fit(features_train, labels_train)
rf_pred = clf.predict(features_test)
rf_score = clf.score(features_test, labels_test)
rf_precision = precision_score(labels_test, rf_pred)
rf_recall = recall_score(labels_test, rf_pred)
print ('Random Forest accuracy:', rf_score)
print ('Random Forest precision:', rf_precision)
print ('Random Forest recall:', rf_recall, '\n')


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
fit = clf.fit(features_train, labels_train)
kn_pred = clf.predict(features_test)
kn_score = clf.score(features_test, labels_test)
kn_precision = precision_score(labels_test, kn_pred)
kn_recall = recall_score(labels_test, kn_pred)
print ('K Neighbors accuracy:', kn_score)
print ('K Neighbors precision:', kn_precision)
print ('K Neighbors recall:', kn_recall, '\n')

### 6. Freature Select

from sklearn.feature_selection import SelectKBest, f_classif

kbest = SelectKBest(f_classif, k=5)
features_selected = kbest.fit_transform(features_train, labels_train)
print(features_selected.shape)

final_features = [features_list[i+1] for i in kbest.get_support(indices=True)]
print(final_features)

from tester import test_classifier

### 7. Chosing GaussianNB model for cross-validation evaluation technique.
from sklearn.metrics import precision_score, recall_score


features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options' ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(features_train, labels_train)
gnb_pred = clf.predict(features_test)
gnb_score = clf.score(features_test, labels_test)
gnb_precision = precision_score(labels_test, gnb_pred)
gnb_recall = recall_score(labels_test, gnb_pred)
print ('GaussianNB accuracy:', gnb_score)
print ('GaussianNB precision:', gnb_precision)
print ('GaussianNB recall:', gnb_recall, '\n')

test_classifier(clf, my_dataset, features_list)

print('------------------------------------------------------------------------------------------------------------')

### Creating a pipelines with hyperparametr tuning for preselected algorythms:
### GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, DeepForest.

### GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tester import test_classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)
select = SelectKBest()
scaler = MinMaxScaler()
gnb = GaussianNB()

steps = [('feature_select', select),
         ('classifier', gnb)]

param_grid = {'feature_select__k': range(1, 17),}
sss = StratifiedShuffleSplit(100, test_size=0.3, random_state=0)
pipe = Pipeline(steps)
gs = GridSearchCV(pipe, param_grid, cv= sss, scoring='f1')

gs.fit(features_train, labels_train)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)

#  Since GaussianNB does'nt have parameters to tune lets try adding and tuning PCA to the pipeline
#  to see the impact of principal components on precision and recall scores.

pca = PCA()
steps = [('scaler', scaler),
         ('dim_red', pca),
         ('feature_select', select),
         ('classifier', gnb)]

param_grid = {
    'feature_select__k': range(1, 17),

    'dim_red__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'dim_red__svd_solver': ['auto', 'full', 'arpack', 'randomized', 'dense'],
    'dim_red__whiten': [True, False],
    'dim_red__random_state': [42],
    'dim_red__iterated_power': [1, 2, 3],

}
feature_select_k_values = list(range(1, 17))

best_PCA_gnb_params = {
    'feature_select__k': feature_select_k_values,

    'dim_red__n_components': [5],
    'dim_red__svd_solver': ['auto'],
    'dim_red__whiten': [True],
    'dim_red__random_state': [42],
    'dim_red__iterated_power': [1]
}
import json
with open('best_PCA_gnb_params.json', 'w') as file:
    json.dump(best_PCA_gnb_params, file)

with open('best_PCA_gnb_params.json', 'r') as file:
    best_PCA_gnb_params = json.load(file)

pipe = Pipeline(steps)
gs = GridSearchCV(pipe, best_PCA_gnb_params, cv=sss, scoring='f1')
gs.fit(features_train, labels_train)
print(gs.best_params_)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)

### KNeighborsClassifier:

knn = KNeighborsClassifier()

steps = [('scaler', scaler),
         ('feature_select', select),
         ('classifier_knn', knn)]

param_grid = {
    'feature_select__k': range(1, 17),

    'classifier_knn__n_neighbors': np.arange(3, 15),
    'classifier_knn__weights': ['uniform', 'distance'],
    'classifier_knn__algorithm': ['ball_tree', 'kd_tree', 'brute']
}
feature_select_knn_values = range(1, 17)
best_params_knn = {
    'feature_select__k': feature_select_knn_values,

    'classifier_knn__n_neighbors': [3],
    'classifier_knn__weights': ['distance'],
    'classifier_knn__algorithm': ['ball_tree']
}


pipe = Pipeline(steps)

gs = GridSearchCV(pipe, best_params_knn, cv=sss, scoring='f1')
gs.fit(features_train, labels_train)
gs.best_params_
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)

### KNeighborsClassifier: hyperparameter tuning

steps = [('scaler', scaler),
         ('feature_select', select),
         ('classifier_knn', knn)]

param_grid = {'feature_select__k': range(1, 17)}

gs = GridSearchCV(pipe, param_grid, cv=sss, scoring='f1')
gs.fit(features_train, labels_train)
gs.best_params_
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)


###  DecisionTreeClassifier:

    #Default settings
dt = DecisionTreeClassifier()
steps = [('feature_select', select),
         ('classifier_dt', dt)]

param_grid = {'feature_select__k': range(1, 17),}

pipe = Pipeline(steps)
gs = GridSearchCV(pipe, param_grid, cv=sss, scoring='f1')
gs.fit(features_train, labels_train)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)

    #Hyperparameter tuning

dt = DecisionTreeClassifier()
steps = [('feature_select', select),
         ('classifier_dt', dt)]

param_grid = {'feature_select__k': range(1, 17),
            'classifier_dt__max_depth': np.arange(3, 10),
            'classifier_dt__min_samples_split': np.arange(3, 10),

              }

pipe = Pipeline(steps)
gs = GridSearchCV(pipe, param_grid, cv=sss, scoring='f1', n_jobs=-1)
gs.fit(features_train, labels_train)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)

###  RandomForestClassifier.

    #   Default settings.

rf = RandomForestClassifier()

steps = [('feature_select', select),
         ('classifier_rf', rf)]
param_grid = {'feature_select__k': range(1,17)}

pipe = Pipeline(steps)
gs = GridSearchCV(pipe, param_grid, cv=sss, scoring='f1', n_jobs=-1)
gs.fit(features_train, labels_train)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)
    # Hyperparameter tuning.


param_grid = {
            'feature_select__k': range(1, 17),
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(10, 100, num=10)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
              }


# sss = StratifiedShuffleSplit(10, test_size=0.3, random_state=42)
# rf_random = RandomizedSearchCV(rf, param_grid, n_iter=100, cv=sss, verbose=2, random_state=42, n_jobs=-1)
# rf_random.fit(features_train,labels_train)
#
# print(rf_random.best_params_)
# clf = rf_random.best_estimator_
# test_classifier(clf, my_dataset, features_list)

steps = [('feature_select', select),
         ('classifier_rf', rf)]

best_param_grid_rf = {
            'feature_select__k': range(1, 17),
            'classifier_rf__n_estimators': [1400],
            'classifier_rf__max_features': ['sqrt'],
            'classifier_rf__max_depth': [40],
            'classifier_rf__min_samples_split': [10],
            'classifier_rf__min_samples_leaf': [1],
            'classifier_rf__bootstrap': [False]
}

pipe = Pipeline(steps)

gs = GridSearchCV(pipe, best_param_grid_rf, cv=sss, scoring='f1', n_jobs=-1)
gs.fit(features_train, labels_train)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

