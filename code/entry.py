import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def readFile(path,y_label,skew_exempted=[], training_ratio=0.7):
    raw = pd.read_csv(path)
    n, d = raw.shape
    training_size = int(n * training_ratio)
    # raw = raw.sample(frac=1).reset_index(drop=True)  # shuffle
    skewed = raw[raw.dtypes[raw.dtypes != "object"].index.drop(skew_exempted)].apply(lambda x: skew(x.dropna()))
    skewed = skewed[skewed > 0.75].index
    # raw[skewed] = np.log1p(raw[skewed])  # reduce skewness
    raw = pd.get_dummies(raw)  # encode categorical features
    raw = raw.fillna(raw.mean())
    train = raw[0:training_size]
    test = raw[training_size:]
    X_train = train.drop(y_label,axis=1)
    X_test = test.drop(y_label,axis=1)
    y_train = train[y_label]
    y_test = test[y_label]
    return X_train, X_test, y_train, y_test

def random_forest(X_train, X_test, y_train, y_test):
    params = {
        'n_estimators': [500],
        'max_features': [5],#[3,5,7,9],
        'max_depth': [140],#[135,140,142]
    }
    model = GridSearchCV(estimator=RandomForestClassifier(max_depth=None), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    print('test error:')
    print(np.mean(model.predict(X_test) != y_test))

def KNN(X_train, X_test, y_train, y_test):
    params = {
        'n_neighbors': [1,2,3]
    }
    model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    print('test error:')
    print(np.mean(model.predict(X_test) != y_test))

def decision_tree(X_train, X_test, y_train, y_test):
    params = {
        'max_depth': [20,40,60,80,100,140,300,600,800,None]
    }
    model = GridSearchCV(estimator=DecisionTreeClassifier(max_depth=None), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    print('test error:')
    print(np.mean(model.predict(X_test) != y_test))

def SVM(X_train, X_test, y_train, y_test):
    params = {
        'C': [0.4,0.6,0.8,1.0,1.2,1.4]
    }
    model = GridSearchCV(estimator=SVC(gamma='scale'), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    print('test error:')
    print(np.mean(model.predict(X_test) != y_test))

def logistic_regression(X_train, X_test, y_train, y_test):
    params = {
        'penalty': ['l1', 'l2'],
        'C': [0.4,0.6,0.8,1.0,1.2,1.4]
    }
    model = GridSearchCV(estimator=LogisticRegression(), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    print('test error:')
    print(np.mean(model.predict(X_test) != y_test))

path = '../data/framingham.csv'
y_label = 'TenYearCHD'
skew_exempted = ['education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD']

X_train, X_test, y_train, y_test = readFile(path,y_label,skew_exempted)
# random_forest(X_train, X_test, y_train, y_test)
# KNN(X_train, X_test, y_train, y_test)
# decision_tree(X_train, X_test, y_train, y_test)
# SVM(X_train, X_test, y_train, y_test)
logistic_regression(X_train, X_test, y_train, y_test)