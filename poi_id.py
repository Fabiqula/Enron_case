#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

#  flatten the input dictionary, and convert it to a list and create df:

data_list = []

for key, value in data_dict.items():
    record = value.copy()
    record['person'] = key
    data_list.append(record)


#  create pandas data frame

df = pd.DataFrame(data_list)

# move person column to pos 0

column_to_move = 'person'
temp_column = df['person']

# Exploring data
df.drop('person', axis=1, inplace=True)
df.insert(0, column_to_move, temp_column)
#
# print(df.head())
# print(sum(df['poi']))

df = df.replace('NaN', np.nan)
# print(df.info())

# Visualising the data
import matplotlib.pyplot as plt
import seaborn as sns

# features = ['salary', 'bonus']
# data = featureFormat(my_dataset, features)

# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     plt.scatter(salary, bonus)
#     plt.title('salary vs bonus')
# # plt.show()

# outlier = max(data, key=lambda x: x[1])
# print(outlier)

# for person, _ in data_dict.items():
#     if data_dict[person]['salary'] == outlier[0]:
#         print(person)


data_dict.pop('TOTAL', 0)


# features = ['salary', 'bonus']
# data = featureFormat(my_dataset, features)

# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     plt.scatter(salary, bonus)
#     plt.title('salary vs bonus')
# # plt.show()

# for person, value in my_dataset.items():
#     if my_dataset[person]['salary'] != 'NaN' and my_dataset[person]['salary'] > 1_000_000 and my_dataset[person]['bonus'] != 'NaN' and my_dataset[person]['bonus'] > 5_000_000:
#         print(f"{person}, salary: {value['salary']}, bonus: {value['bonus']}")

### Task 3: Create new feature(s)
for person in my_dataset:
    if my_dataset[person]['from_poi_to_this_person'] != 'NaN' and my_dataset[person]['from_this_person_to_poi'] != 'NaN':
        my_dataset[person]['from_poi_ratio'] = my_dataset[person]['from_poi_to_this_person']/my_dataset[person]['to_messages']
        my_dataset[person]['to_poi_ratio'] = my_dataset[person]['from_this_person_to_poi']/my_dataset[person]['to_messages']

    else:
        my_dataset[person]['from_poi_ratio'] = 'NaN'
        my_dataset[person]['to_poi_ratio'] = 'NaN'

# features = ['poi','from_poi_ratio', 'to_poi_ratio']
# data = featureFormat(my_dataset, features)

# for person in data:
#     from_poi_point = person[1]
#     to_poi_point = person[2]
#     if person[0] == 0:
#         plt.scatter(from_poi_point, to_poi_point, color='black', alpha=0.4)
#     else:
#         plt.scatter(from_poi_point, to_poi_point, color='red')

# plt.xlabel('from_poi_point')
# plt.ylabel('to_poi_point')
# plt.show()


### Extract features and labels from dataset for local testing
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'from_poi_ratio', 'to_poi_ratio']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
gnb_pred = clf.predict(features_test)
gnb_score = clf.score(features_test, labels_test)
gnb_precision = precision_score(labels_test, gnb_pred)
gnb_recall = recall_score(labels_test, gnb_pred)
gnb_f1 = f1_score(labels_test, gnb_pred)
print ('GaussianNB accuracy:', gnb_score)
print ('GaussianNB precision:', gnb_precision)
print ('GaussianNB recall:', gnb_recall)
print ('GaussianNB f1:', gnb_f1, '\n')

#
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)
# fit = clf.fit(features_train, labels_train)
# dt_pred = clf.predict(features_test, labels_test)
# dt_score = clf.score(features_test, labels_test)
# dt_precision = precision_score(labels_test, dt_pred)
# dt_recall = recall_score(labels_test, dt_pred)
# dt_f1 = f1_score(labels_test, dt_pred)
# print ('Decision Tree accuracy:', dt_score)
# print ('Decision Tree precision:', dt_precision)
# print ('Decision Tree recall:', dt_recall)
# print ('Decision Tree f1:', dt_f1, '\n')
#
#
# from sklearn.svm import SVC
# clf = SVC(random_state=0)
# fit = clf.fit(features_train, labels_train)
# svc_pred = clf.predict(features_test)
# svc_score = clf.score(features_test, labels_test)
# svc_precision = precision_score(labels_test, svc_pred)
# svc_recall = recall_score(labels_test, svc_pred)
# svc_f1 = f1_score(labels_test, svc_pred)
# print ('SVC accuracy:', svc_score)
# print ('SVC precision:', svc_precision)
# print ('SVC recall:', svc_recall)
# print ('SVC f1:', svc_f1, '\n')
#
#
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(random_state=0)
# fit = clf.fit(features_train, labels_train)
# rf_pred = clf.predict(features_test)
# rf_score = clf.score(features_test, labels_test)
# rf_precision = precision_score(labels_test, rf_pred)
# rf_recall = recall_score(labels_test, rf_pred)
# rf_f1 = f1_score(labels_test, rf_pred)
# print ('Random Forest accuracy:', rf_score)
# print ('Random Forest precision:', rf_precision)
# print ('Random Forest recall:', rf_recall)
# print ('Random Forest f1:', rf_f1, '\n')
#
#
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()
# fit = clf.fit(features_train, labels_train)
# kn_pred = clf.predict(features_test)
# kn_score = clf.score(features_test, labels_test)
# kn_precision = precision_score(labels_test, kn_pred)
# kn_recall = recall_score(labels_test, kn_pred)
# kn_f1 = f1_score(labels_test, kn_pred)
# print ('K Neighbors accuracy:', kn_score)
# print ('K Neighbors precision:', kn_precision)
# print ('K Neighbors recall:', kn_recall)
# print ('K Neighbors kn_f1:', kn_f1, '\n')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.feature_selection import SelectKBest, f_classif

kbest = SelectKBest(f_classif,k=6)
features_selected = kbest.fit_transform(features_train, labels_train)
# print(features_selected.shape)
# print(kbest.get_support(indices=True))

final_features = [features_list[i+1] for i in kbest.get_support(indices=True)]
# print(final_features)


# Example starting point. Try investigating other evaluation techniques!
features_selected = ['poi', 'salary', 'bonus', 'total_payments', 'total_stock_value',
                     'exercised_stock_options', 'shared_receipt_with_poi']
data = featureFormat(my_dataset, features_selected)
labels, features = targetFeatureSplit(data)


features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
gnb_pred = clf.predict(features_test)
gnb_score = clf.score(features_test, labels_test)
gnb_precision = precision_score(labels_test, gnb_pred)
gnb_recall = recall_score(labels_test, gnb_pred)
gnb_f1 = f1_score(labels_test, gnb_pred)
print ('GaussianNB accuracy:', gnb_score)
print ('GaussianNB precision:', gnb_precision)
print ('GaussianNB recall:', gnb_recall)
print ('GaussianNB f1:', gnb_f1, '\n')


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
fit = clf.fit(features_train, labels_train)
dt_pred = clf.predict(features_test, labels_test)
dt_score = clf.score(features_test, labels_test)
dt_precision = precision_score(labels_test, dt_pred)
dt_recall = recall_score(labels_test, dt_pred)
dt_f1 = f1_score(labels_test, dt_pred)
print ('Decision Tree accuracy:', dt_score)
print ('Decision Tree precision:', dt_precision)
print ('Decision Tree recall:', dt_recall)
print ('Decision Tree f1:', dt_f1, '\n')


from sklearn.svm import SVC
clf = SVC(random_state=0)
fit = clf.fit(features_train, labels_train)
svc_pred = clf.predict(features_test)
svc_score = clf.score(features_test, labels_test)
svc_precision = precision_score(labels_test, svc_pred)
svc_recall = recall_score(labels_test, svc_pred)
svc_f1 = f1_score(labels_test, svc_pred)
print ('SVC accuracy:', svc_score)
print ('SVC precision:', svc_precision)
print ('SVC recall:', svc_recall)
print ('SVC f1:', svc_f1, '\n')


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
fit = clf.fit(features_train, labels_train)
rf_pred = clf.predict(features_test)
rf_score = clf.score(features_test, labels_test)
rf_precision = precision_score(labels_test, rf_pred)
rf_recall = recall_score(labels_test, rf_pred)
rf_f1 = f1_score(labels_test, rf_pred)
print ('Random Forest accuracy:', rf_score)
print ('Random Forest precision:', rf_precision)
print ('Random Forest recall:', rf_recall)
print ('Random Forest f1:', rf_f1, '\n')


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
fit = clf.fit(features_train, labels_train)
kn_pred = clf.predict(features_test)
kn_score = clf.score(features_test, labels_test)
kn_precision = precision_score(labels_test, kn_pred)
kn_recall = recall_score(labels_test, kn_pred)
kn_f1 = f1_score(labels_test, kn_pred)
print ('K Neighbors accuracy:', kn_score)
print ('K Neighbors precision:', kn_precision)
print ('K Neighbors recall:', kn_recall)
print ('K Neighbors kn_f1:', kn_f1, '\n')

### Tuning, validating and evaluating metrics

features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'from_poi_ratio', 'to_poi_ratio']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier

scaler = MinMaxScaler()
select = SelectKBest()
gnb = GaussianNB()

steps = [('scaler', scaler),
         ('feature_select', select),
         ('classifier', gnb)]

param_grid = {'feature_select__k': range(1,17)}
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
pipe = Pipeline(steps)
gs = GridSearchCV(pipe, param_grid, cv=sss, scoring='f1')

gs.fit(features_train, labels_Śtrain)
clf = gs.best_estimator_

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)