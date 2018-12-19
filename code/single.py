import numpy as np
from fileIO import readFile
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

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
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

def KNN(X_train, X_test, y_train, y_test):
    params = {
        'n_neighbors': [2], # [1,2,3]
    }
    model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

def decision_tree(X_train, X_test, y_train, y_test):
    params = {
        'max_depth': [1], # [1,2,3,4,5,6,10,15,20,60,None]
    }
    model = GridSearchCV(estimator=DecisionTreeClassifier(max_depth=None), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

def SVM(X_train, X_test, y_train, y_test):
    params = {
        'C': [0.1], # [0.1,0.2,0.3,0.4,0.6,0.8]
        'kernel': ['rbf'],
    }
    # model = GridSearchCV(estimator=SVC(gamma='scale'), param_grid=params, cv=10)
    model = SVC(gamma='scale', kernel='rbf')
    model.fit(X_train, y_train)

    # print('best params:')
    # print(model.best_params_)
    
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

def logistic_regression(X_train, X_test, y_train, y_test):
    params = {
        'penalty': ['l1'], # ['l1', 'l2'],
        'C': [1.0], # [0.4,0.6,0.8,1.0,1.2,1.4]
    }
    model = GridSearchCV(estimator=LogisticRegression(), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

def gradient_boosting(X_train, X_test, y_train, y_test):
    params = {
        'max_depth': [3], # [1, 2, 3, 5, 7],
        'learning_rate': [0.03], # [0.01, 0.03, 0.05, 0.1, 0.2],
        'n_estimators': [100], # [75, 100, 125, 150],
    }
    model = GridSearchCV(estimator=XGBClassifier(), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

def NN(X_train, X_test, y_train, y_test):
    params = {
        'hidden_layer_sizes': [(100)], # [(100), (50,50), (20,20,20)],
        'solver': ['lbfgs'], # ['lbfgs', 'adam'],
        'alpha': [0.001], # [0.0001, 0.001, 0.01, 0.1],
    }
    model = GridSearchCV(estimator=MLPClassifier(batch_size=100), param_grid=params, cv=10)
    model.fit(X_train, y_train)
    print('best params:')
    print(model.best_params_)
    print('train error:')
    print(np.mean(model.predict(X_train) != y_train))
    y_hat = model.predict(X_test)
    print('test error:')
    print(np.mean(y_hat != y_test))
    print('Sen:')
    print(np.mean(y_test[y_hat==1]))
    print('Spe:')
    print(np.mean(y_test[y_hat==0] == 0))
    print('Pa:')
    print(np.mean(y_hat == y_test))

# path = '../data/framingham.csv'
# y_label = 'TenYearCHD'
# encode_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
# skew_exempted = ['education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD']

# X_train, X_test, y_train, y_test = readFile(path=path, y_label=y_label, encode_features=encode_features, skew_exempted=skew_exempted)

path = '../data/salary.csv'
y_label = 'salary'
encode_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

X_train, X_test, y_train, y_test = readFile(path=path, y_label=y_label, encode_features=encode_features, training_ratio=0.7)

# random_forest(X_train, X_test, y_train, y_test)
# KNN(X_train, X_test, y_train, y_test)
# decision_tree(X_train, X_test, y_train, y_test)
# SVM(X_train, X_test, y_train, y_test)
# logistic_regression(X_train, X_test, y_train, y_test)
# gradient_boosting(X_train, X_test, y_train, y_test)
NN(X_train, X_test, y_train, y_test)