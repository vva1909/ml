import numpy as np

def mse(y_test, y_pred):
    return (((y_pred - y_test) ** 2).sum()).mean()

def rmse(y_test, y_pred):
    return np.sqrt(np.mean((y_test - y_pred) ** 2))

def mae(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))

def r2(y_test, y_pred):
    return 1 - (((y_pred - y_test) ** 2).sum()) / (((y_test - y_test.mean()) ** 2).sum())