import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, epochs, learning_rate, bias=True, lambda_=0.0001):
        self.epochs = epochs
        self.lr = learning_rate
        self.bias = bias
        self.lambda_ = lambda_
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def prob(self, X, type_=False):
        if not type_:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return self.sigmoid(X @ self.weights)

    def gradient(self, X, y):
        a = self.prob(X, True)
        return X.T @ (a - y) / X.shape[0] + self.lambda_ * self.weights

    def loss(self, X, y, type=False):
        a = self.prob(X, type)
        loss_0 = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
        weight_decay = 0.5 * self.lambda_ / X.shape[0] * np.sum(self.weights ** 2)
        return loss_0 + weight_decay

    def fit(self, X, y):
        if self.bias:
            X = np.column_stack([np.ones(X.shape[0]), X])

        self.weights = np.random.randn(X.shape[1])
        for _ in range(self.epochs):
            grad = self.gradient(X, y)
            self.weights -= self.lr * grad

    def predict(self, X, type=False):
        return (self.prob(X, type) > 0.5).astype(int)