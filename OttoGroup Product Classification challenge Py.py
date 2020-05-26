# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:06:10 2020

@author: jaket
"""

#%% import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#%% set wd, load data

os.chdir(r"C:/Users/jaket/Dropbox/Kaggle/OttoGroup_product_classification")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%% inspect

train.isnull().sum() #no NAs
train.describe()


#since the variables in this are meaningless, its simply a ML problem with little EDA, vis/feature engineering etc.

#%% set up ML
from sklearn.model_selection import train_test_split

train.target = train['target'].astype('category')

#get features, get labels, split train/test
X = np.array(train.drop(['target', 'id'], axis=1))
y= np.array(train['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Import the three supervised learning models from sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, log_loss, recall_score, f1_score
from xgboost import XGBClassifier



#from keras import Sequential
#from keras.layers import Dense

#%% do ML, compare methods
classifiers = [
    KNeighborsClassifier(3),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

# loop over every classifier algo
for clf in classifiers:
    print("="*30)

    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    
    # calculate score
    precision = precision_score(y_test, train_predictions, average = 'macro') 
    recall = recall_score(y_test, train_predictions, average = 'macro') 
    f_score = f1_score(y_test, train_predictions, average = 'macro')
    
    
    print("Precision: {:.4%}".format(precision))
    print("Recall: {:.4%}".format(recall))
    print("F-score: {:.4%}".format(recall))
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

#%% plot training results for algo determination

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')

# The metrics suggest that XGB, KNN and DTC are the best performing metrics.
# Next we will tune these

#%% parameter tuning - XGB
from sklearn.model_selection import GridSearchCV

parameters_xgb = {
        'learning_rate': [0.05, 0.2],
        'n_estimators': [200, 400], 
        'max_depth': [5,10],
        'gamma' :[0.1, 1],       
        'subsample': [0.5, 0.75],
        'colsample_bytree': [0.5,1],
        }

xgb = XGBClassifier(objective='multi:softprob',silent=False)

# Instantiate the grid search model
grid_search_xgb = GridSearchCV(estimator = xgb, param_grid = parameters_xgb, 
                          cv = 3,n_jobs=-1, verbose = 2)

# Fit the grid search to the data
grid_search_xgb.fit(X_train,y_train)
grid_search_xgb.best_params_

#run best params
best_grid_xgb = grid_search_xgb.best_estimator_
best_grid_xgb.fit(X_train,y_train)

#predict test data
test = np.array(train.drop(['id'], axis=1))
test = 
y_pred_xgb = best_grid_xgb.predict(test)

test.columns
X_train.columns
# calculate score
precision = precision_score(y_test, y_pred, average = 'macro') * 100
recall = recall_score(y_test, y_pred, average = 'macro') * 100
f_score = f1_score(y_test, y_pred, average = 'macro') * 100
a_score = accuracy_score(y_test, y_pred) * 100

print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F-Measure: %.3f' % f_score)
print('Accuracy: %.3f' % a_score)

#%% parameter tuning - KNN
knn = KNeighborsClassifier()

parameters_knn = {'n_neighbors':[5,7, 10, 20, 50], 
          'leaf_size':[0.001, 0.1,1, 3, 5], 
          'weights':['distance', 'uniform'], 
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}

# Instantiate the grid search model
grid_search_knn = GridSearchCV(estimator = knn, param_grid = parameters_knn, 
                          cv = 3,  verbose = 2, n_jobs=2)

# Fit the grid search to the data
grid_search_knn.fit(X_train,y_train)
grid_search_knn.best_params_

#run best params
best_grid_knn = grid_search_knn.best_estimator_
best_grid_knn.fit(X_train,y_train)
y_pred_knn=best_grid_knn.predict(test)

# calculate score
precision = precision_score(y_test, y_pred, average = 'macro') * 100
recall = recall_score(y_test, y_pred, average = 'macro') * 100
f_score = f1_score(y_test, y_pred, average = 'macro') * 100
a_score = accuracy_score(y_test, y_pred) * 100

print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F-Measure: %.3f' % f_score)
print('Accuracy: %.3f' % a_score)
#small increases in precision and accuracy with tuning the RF.

#%% ensemble
from sklearn import ensemble
ensemble_clf=ensemble.VotingClassifier([('clf_knn',best_grid_knn),('clf_xgb',best_grid_xgb)],voting='soft',weights=[1,1]);


#%% check submission format
from sklearn.preprocessing import OneHotEncoder
OHE=OneHotEncoder()

submission_example=pd.read_csv('sampleSubmission.csv')

y_pred_knn=pd.DataFrame(y_pred_knn)
y_pred_knn.columns=['class']
y_pred_knn['class']=y_pred_knn['class'].astype('category') 
sub_knn=pd.DataFrame(OHE.fit_transform(y_pred_knn[['class']]).toarray())
sub_knn.columns=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
sub_knn['id'] = sub_knn.index
sub_knn.to_csv('knn_otto_submission.csv', index=False)

y_pred_xgb=pd.DataFrame(y_pred_xgb)
y_pred_xgb.columns=['class']
y_pred_xgb['class']=y_pred_xgb['class'].astype('category') 
sub_xgb=pd.DataFrame(OHE.fit_transform(y_pred_xgb[['class']]).toarray())
sub_xgb.columns=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
sub_xgb['id'] = sub_xgb.index
sub_xgb.to_csv('xgb_otto_submission.csv', index=False)

y_pred_xgb.shape

