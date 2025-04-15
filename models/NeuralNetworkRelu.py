import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

class NeuralNetworkReLU:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha  
        self.W = []
        self.b = []

        # He initialization for ReLU
        for i in range(0, len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0/layers[i])
            b_ = np.zeros((1, layers[i+1]))
            self.W.append(w_)
            self.b.append(b_)

    def fit_partial(self, X, y):
        X = np.array(X).reshape(-1, self.layers[0])
        y = np.array(y).reshape(-1, self.layers[-1])
        
        A = [X]
        # feedforward with ReLU
        out = A[-1]
        for i in range(0, len(self.layers) - 2):  # All hidden layers use ReLU
            out = relu(np.dot(out, self.W[i]) + self.b[i])
            A.append(out)
        
        # Output layer uses sigmoid for binary classification
        out = sigmoid(np.dot(out, self.W[-1]) + self.b[-1])
        A.append(out)

        # backpropagation
        dA = [-(y/A[-1]) + ((1-y)/(1-A[-1]))]  # Output layer gradient
        dW = []
        db = []

        # Output layer (sigmoid)
        dw_ = np.dot(A[-2].T, dA[-1] * sigmoid_derivative(A[-1]))
        db_ = np.sum(dA[-1] * sigmoid_derivative(A[-1]), axis=0, keepdims=True)
        dA_ = np.dot(dA[-1] * sigmoid_derivative(A[-1]), self.W[-1].T)
        dW.append(dw_)
        db.append(db_)
        dA.append(dA_)

        # Hidden layers (ReLU)
        for i in reversed(range(0, len(self.layers)-2)):
            dw_ = np.dot(A[i].T, dA[-1] * relu_derivative(A[i+1]))
            db_ = np.sum(dA[-1] * relu_derivative(A[i+1]), axis=0, keepdims=True)
            dA_ = np.dot(dA[-1] * relu_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        dW = dW[::-1]
        db = db[::-1]

        # Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))

    def predict(self, X):
        X = np.array(X).reshape(-1, self.layers[0])
        # Hidden layers with ReLU
        for i in range(0, len(self.layers) - 2):
            X = relu(np.dot(X, self.W[i]) + self.b[i])
        # Output layer with sigmoid
        X = sigmoid(np.dot(X, self.W[-1]) + self.b[-1])
        return X

    def calculate_loss(self, X, y):
        y_pred = self.predict(X)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))