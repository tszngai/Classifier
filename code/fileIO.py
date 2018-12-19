import pandas as pd
import numpy as np
from scipy.stats import skew

def readFile(path, y_label, encode_features=[], skew_exempted=[], training_ratio=0.7, shuffle=True, needSkew=False):
    raw = pd.read_csv(path)
    n, d = raw.shape
    training_size = int(n * training_ratio)

    if (shuffle):
        raw = raw.sample(frac=1).reset_index(drop=True)  # shuffle
    
    if (needSkew):
        skewed = raw[raw.dtypes[raw.dtypes != "object"].index.drop(skew_exempted)].apply(lambda x: skew(x.dropna()))
        skewed = skewed[skewed > 0.75].index
        raw[skewed] = np.log1p(raw[skewed])  # reduce skewness
    
    raw = pd.get_dummies(raw, columns=encode_features)  # encode categorical features
    raw = raw.fillna(raw.mean())
    train = raw[0:training_size]
    test = raw[training_size:]
    X_train = train.drop(y_label,axis=1)
    X_test = test.drop(y_label,axis=1)
    y_train = train[y_label]
    y_test = test[y_label]
    return X_train, X_test, y_train, y_test