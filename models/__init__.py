# models/__init__.py
from .LinearRegression import LinearRegression, LassoRegression, RidgeRegression
from .Evaluate import r2, mae, rmse, mse
from .PreProcessing import my_train_test_split
# import os
# os.chdir('D:/nah/MachineLearning')
# print(os.getcwd())
#
# import sys
# print(sys.path)