# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:08:28 2018

@author: Zoltan
"""

import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import set_option

import numpy as np
from numpy import set_printoptions

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, auc

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout


def plot_correlation_matrix(correlations, column_names):
    # plot correlation matrix
    # cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", ["blue"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap=plt.cm.gray)
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(column_names)
    ax.set_yticklabels(column_names)
    plt.show()
    
    
def get_redundant_pairs(df):
    #Get diagonal and lower triangular pairs of correlation matrix
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def plot_ROC(predictions_test, outcome_test):
    fpr, tpr, thresholds = roc_curve(predictions_test, outcome_test)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

arrythmia_dataset = 'arrhythmia.csv';
cleveland_dataset = 'processed.cleveland.csv';

# Read csv file into a pandas dataframe
df = pd.read_csv(cleveland_dataset)

# Name of the label colum
label_column = 'label'

# Column names
names = np.array(df.columns)
print(names)

# Take a look at the first few rows
print(df.head())

# Print the names of the columns
print(df.columns)

# Preprocess dataset, convert label 1,2,3 to 1 and 0 remains 0
df[label_column] = [x if x == 0 else 1 for x in df[label_column]]
print(df[label_column])

# Print out the available classes
print(df[label_column])
positive_class_count = df.loc[df[label_column]==1, label_column].size
negative_class_count = df.loc[df[label_column]==0, label_column].size

# Label distribution
print(positive_class_count)
print(negative_class_count)

# convert ? t0 NA
for i in df.columns:
    df[i] = [x if x != '?' else 0 for x in df[i]]
    
# check is there are ? remained
for i in df.columns:
    for j in df[i]:
        if (j == '?'):
            print(j)

# print some statistical characteristics
pd.set_option('display.max_columns', None)
print(df.describe())

# box and whisker plots
df.boxplot()

# histogramm of all attributes
df.hist()

# scatter matrix of all attributes
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

# create corelation matrix
set_option('display.width', 100)
set_option('precision', 3)

correlations = df.corr(method='pearson')
print(correlations)

# top correlations
s = correlations.unstack()
so = s.sort_values(kind="quicksort")

# plot correlation matrix
plot_correlation_matrix(correlations, names)

# top 5 correlations
# top_3 = get_top_abs_correlations(df, 3)

# top 5 correlations
# top_3 = get_top_abs_correlations(df, 3)

# verify the skewness of the attributes
skew = df.skew()
print(skew)

# Normalize because the standard deviation of some attributes (cholesterole, maximum heart rate - talach) 
# is too high compared to age or oldpeak
array = df.values
array_len = len(array[0])
X = array[:,0:array_len-1]
Y = array[:,-1]
Y = Y.astype('int')

normalizer = Normalizer().fit(X)
normalizedX = normalizer.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

# TODO Feature selection

# Feature importance
normalizer = ExtraTreesClassifier()
normalizer.fit(normalizedX, Y)
print(normalizer.feature_importances_)

# train and test
num_folds = 10
seed = 7

scoring_accuracy = 'accuracy'
scoring_negative_log_loss = 'neg_log_loss'
scoring_ROC = 'roc_auc'
scoring_MAE = 'neg_mean_absolute_error'
scoring_MSE = 'neg_mean_squared_error'
scoring_R2 = 'r2'

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, normalizedX, Y, cv=kfold, scoring=scoring_ROC)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# prepare models
models = []
models.append(('LR', LogisticRegression(penalty='l2', solver='newton-cg')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(C=1.33)))
# models.append(('RIDGE', Ridge(alpha=0.97)))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, rescaledX, Y, cv=kfold, scoring=scoring_ROC)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Boosting
num_trees = 30
seed=7
kfold = KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring_accuracy)
print(results.mean())

# Stochastic Gradient Boosting
seed = 7
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Voting
kfold = KFold(n_splits=10, random_state=7)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring_accuracy)
print(results.mean())


# Voting
kfold = KFold(n_splits=10, random_state=7)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = LinearDiscriminantAnalysis()
estimators.append(('LDA', model2))
model3 = GaussianNB()
estimators.append(('NB', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring_accuracy)
print(results.mean())

# Parameter tuning
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

param_grid = {'alpha': uniform()}
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100,
random_state=7)
rsearch.fit(X, Y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)

# nural network sklearn
model = MLPClassifier(hidden_layer_sizes=(100, 1000, 100), max_iter=1000)
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(model, normalizedX, Y, cv=kfold, scoring=scoring_ROC)
results.append(cv_results)
msg = "%s: %f (%f)" % ("NN", cv_results.mean(), cv_results.std())
print(msg)

# recurrent neural network











