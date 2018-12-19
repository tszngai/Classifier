import numpy as np
from single import XGBClassifier, RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier, SVC, LogisticRegression, MLPClassifier, GridSearchCV
from fileIO import readFile

models = []
models.append(RandomForestClassifier(n_estimators=500, max_features=5, max_depth=140))
models.append(KNeighborsClassifier(n_neighbors=1))
models.append(KNeighborsClassifier(n_neighbors=2))
models.append(DecisionTreeClassifier(max_depth=1))
models.append(SVC(gamma='scale', C=0.1))
models.append(LogisticRegression(penalty='l1', C=1.0))
models.append(XGBClassifier(max_depth=3, learning_rate=0.03, n_estimators=100))
models.append(MLPClassifier(batch_size=100, hidden_layer_sizes=(100), solver='lbfgs', alpha=0.001))

def formEnsemble(X, y=None):
    n, _ = X.shape
    Z = np.zeros((n,len(models)))

    for model in models:
        if y is not None:
            model.fit(X, y)
        Z[:, models.index(model)] = model.predict(X)
    
    return Z

def fitEnsemble(Z, y):
    params = {
        'penalty': ['l1', 'l2'],
        'C': [0.4,0.6,0.8,1.0,1.2,1.4]
    }
    model_stack = GridSearchCV(estimator=LogisticRegression(), param_grid=params, cv=10)
    model_stack.fit(Z, y)
    print('best params:')
    print(model_stack.best_params_)
    print('training error:')
    print(np.mean(model_stack.predict(Z) != y))
    return model_stack

# path = '../data/framingham.csv'
# y_label = 'TenYearCHD'
# encode_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
# skew_exempted = ['education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD']

# X_train, X_test, y_train, y_test = readFile(path=path, y_label=y_label, encode_features=encode_features, skew_exempted=skew_exempted)

path = '../data/salary.csv'
y_label = 'salary'
encode_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

X_train, X_test, y_train, y_test = readFile(path=path, y_label=y_label, encode_features=encode_features, training_ratio=0.7)

# y_train[y_train == 0] = -1
# y_test[y_test == 0] = -1

print('Starting forming ensemble matrix for training set..')
Z_train = formEnsemble(X_train, y_train)
print('Ensemble matrix formed')

print('Starting fitting ensemble matrix for training set..')
model_stack = fitEnsemble(Z_train, y_train)

print('Starting forming ensemble matrix for testing set..')
Z_test = formEnsemble(X_test)
print('Ensemble matrix formed')

y_hat = model_stack.predict(Z_test)
print('test error:')
print(np.mean(y_hat != y_test))
print('Sen:')
print(np.mean(y_test[y_hat==1]))
print('Spe:')
print(np.mean(y_test[y_hat==0] == 0))
print('Pa:')
print(np.mean(y_hat == y_test))

# print(model_stack.predict(Z_test))
# print('true:')
# print(y_test)
# print(np.sum(model_stack.predict(Z_test)))
# print(np.sum(y_test))
# print(np.sum(model_stack.predict(Z_test) != y_test))