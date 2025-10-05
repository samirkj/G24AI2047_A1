import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def load_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
    target = raw.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_features_target(df, target_col='MEDV'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('model', model)])

def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_pipeline(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return float(mse)

def cross_val_mse(pipeline, X, y, cv=5):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    return float((-scores).mean())