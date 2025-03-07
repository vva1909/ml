import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, bias=True):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = bias

    def predict(self, X):
        return X @ self.weights

    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum())

    def gradient(self, X, y):
        y_pred = self.predict(X)
        error = y - y_pred
        if self.bias:
            X = np.column_stack([np.ones(X.shape[0]), X])
        n = X.shape[0]
        grad = (-2.0 / n) * (X.T @ error)
        return grad

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.bias:
            self.weights = np.zeros(n_features + 1)
        else:
            self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):
            grad = self.gradient(X, y)
            self.weights -= self.lr * grad


class LassoRegression(LinearRegression):
    def __init__(self, learning_rate=0.01, epochs=1000, bias=True, lambda_ = 0.1):
        super().__init__(learning_rate, epochs, bias)
        self.lambda_ = lambda_

    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum()) + self.lambda_ * (abs(self.weights)).sum()

    def gradient(self, X, y):
        y_pred = self.predict(X)
        error = y - y_pred
        if self.bias:
            X = np.column_stack([np.ones(X.shape[0]), X])
        n = X.shape[0]
        grad = (-2.0 / n) * (X.T @ error) + self.lambda_ * np.sign(self.weights)
        return grad

class RidgeRegression(LinearRegression):
    def __init__(self, learning_rate=0.01, epochs=1000, bias=True, lambda_=0.1):
        super().__init__(learning_rate, epochs, bias)
        self.lambda_ = lambda_

    def loss_function(self, y_true, y_pred):
        n = y_true.shape[0]
        return (1.0 / n) * (((y_true - y_pred) ** 2).sum()) + self.lambda_ * (self.weights ** 2).sum()

    def gradient(self, X, y):
        y_pred = self.predict(X)
        error = y - y_pred
        if self.bias:
            X = np.column_stack([np.ones(X.shape[0]), X])
        n = X.shape[0]
        grad = (-2.0 / n) * (X.T @ error) + 2 * self.lambda_ * self.weights
        return grad


