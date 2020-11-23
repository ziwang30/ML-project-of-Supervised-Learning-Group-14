#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('......./covtype.csv')
df = df.dropna()
df.head()

print('Number of Records:', df.shape[0])
print('Number of Features:', df.shape[1])


# Data balancing
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='all',
                         random_state=0)  # Resample all classes but the minority class (type 4)
X = df.iloc[:, :-1].values
y = df.iloc[:, 54].values
X_resampled, y_resampled = rus.fit_resample(X, y)

# 'majority': resample only the majority class;
# 'not minority': resample all classes but the minority class;
# 'not majority': resample all classes but the majority class;
# 'all': resample all classes;
# 'auto': equivalent to 'not minority'.

from collections import Counter
print(X_resampled.size)
print(y_resampled.size)
Counter(y_resampled)

# Data Extraction
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, stratify=y_resampled,
                                                    train_size=18900,test_size=329,
                                                    random_state=0)

# cross validation
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, random_state=0)
for train_index, test_index in sss.split(X_train,y_train):
    #print('train_index', train_index, 'test_index', test_index)
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]

# Data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

#############################KNN###############################################

# Hyperparameter model of KNN
best_k = 0
best_score = 0.0
best_weightofknn = ''
for method in ["uniform","distance"]:
    for k in range(1,20):
        knnclassifier = KNeighborsClassifier(n_neighbors = k,weights = method)
        knnclassifier.fit(train_X,train_y)
        score = knnclassifier.score(test_X,test_y)
        if score > best_score:
            best_score = score
            best_k = k
            best_weightofknn = method
print("best_k = %s" %(best_k))
print("best_score = %s"%(best_score))
print("best_weightofknn = %s"%(best_weightofknn))


# KNN model traing
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=best_k,weights=best_weightofknn)
clf=classifier.fit(train_X, train_y)
y_pred = classifier.predict(test_X)

print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(clf.score(train_X, train_y)))
print('Accuracy of KNN classifier on test set: {:.2f}'
     .format(clf.score(test_X, test_y)))


# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(test_y, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(test_y, y_pred)
print("Classification Report:",)
print (result1)


######################################## ANN############################

# Hyperparameter model of ANN

best_score = 0.0
best_activation = ''
best_weightoptimization=''
best_learningrate=''

for activations in ["identity", "logistic", "tanh", "relu"]:
    for solvers in ["lbfgs", "sgd", "adam"]:
        for learningrates in ["constant","invscaling", "adaptive"]:

            annclassifier = MLPClassifier(hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), activation= activations,
                                          solver= solvers, learning_rate= learningrates, random_state=0)
            annclassifier.fit(train_X,train_y)
            score = annclassifier.score(test_X,test_y)
            if score > best_score:
                best_score = score
                best_activation = activations
                best_weightoptimization = solvers
                best_learningrate = learningrates

print("best_score = %s"%(best_score))
print("best_activation = %s"%(best_activation))
print("best_weightofann = %s"%(best_weightoptimization))
print("best_learningrate = %s"%(best_learningrate))

# ANN model traing
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(60,60,60,60,60,60,60), activation= best_activation,
                    solver= best_weightoptimization,learning_rate= best_learningrate, random_state=0,
                    ).fit(train_X,train_y)

y_pred = clf.predict(test_X)

print('Accuracy of ANN classifier on training set: {:.2f}'
      .format(clf.score(train_X, train_y)))
print('Accuracy of ANN classifier on test set: {:.2f}'
      .format(clf.score(test_X, test_y)))


# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(test_y, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(test_y, y_pred)
print("Classification Report:", )
print(result1)
