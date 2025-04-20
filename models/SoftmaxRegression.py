from __future__ import print_function
import numpy as np

import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.loss_hist = None

    def softmax_stable(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A = e_Z / e_Z.sum(axis=1, keepdims=True)
        return A

    def softmax_loss(self, X, y):
        A = self.softmax_stable(X.dot(self.W))
        id0 = range(X.shape[0])
        return -np.mean(np.log(A[id0, y]))

    def softmax_grad(self, X, y):
        A = self.softmax_stable(X.dot(self.W))
        id0 = range(X.shape[0])
        A[id0, y] -= 1
        return X.T.dot(A) / X.shape[0]

    def fit(self, X, y):
        self.W = np.random.randn(X.shape[1], len(np.unique(y))) * 0.01
        N = X.shape[0]
        nbatches = int(np.ceil(float(N) / self.batch_size))
        loss_hist = []
        for ep in range(self.epochs):
            mix_ids = np.random.permutation(N)
            for i in range(nbatches):
                batch_ids = mix_ids[self.batch_size * i:min(self.batch_size * (i + 1), N)]
                X_batch, y_batch = X[batch_ids], y[batch_ids]
                self.W -= self.lr * self.softmax_grad(X_batch, y_batch)

            loss_hist.append(self.softmax_loss(X, y))
        return self.W, loss_hist

    def predict(self, X):
        A = self.softmax_stable(X.dot(self.W))
        return np.argmax(A, axis=1)
