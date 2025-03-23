import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, epochs, learning_rate, bias=True, lambda_=0.0001):
        self.epochs = epochs
        self.lr = learning_rate
        self.bias = bias
        self.lambda_ = lambda_
        self.weights = None

    def prob(self, X):
        if X.shape[1] != self.weights.shape[0]:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return sigmoid(X @ self.weights)

    def gradient(self, X, y):
        a = self.prob(X)
        return X.T @ (a - y) + self.lambda_ * self.weights

    def loss(self, X, y):
        a = self.prob(X)
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

    def predict(self, X):
        return (self.prob(X) > 0.5).astype(int)


    