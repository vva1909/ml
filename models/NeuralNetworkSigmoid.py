import numpy as np

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

class NeuralNetworkSigmoid:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha  
        self.W = []  # Sửa từ w thành W
        self.b = []

        # Khởi tạo weights với He initialization
        for i in range(0, len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((1, layers[i+1]))  # Sửa shape của bias
            self.W.append(w_)
            self.b.append(b_)

    def fit_partial(self, X, y):
        A = [X]
        # feedforward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + self.b[i])
            A.append(out)

        # backpropagation
        X = np.array(X).reshape(-1, self.layers[0])
        y = np.array(y).reshape(-1, self.layers[-1])
        dA = [-(y/A[-1]) + ((1-y)/(1-A[-1]))]
        dW = []
        db = []

        for i in reversed(range(0, len(self.layers)-1)):
            dw_ = np.dot(A[i].T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = np.sum(dA[-1] * sigmoid_derivative(A[i+1]), axis=0, keepdims=True)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        dW = dW[::-1]
        db = db[::-1]

        # Gradient descent với gradient clipping
        for i in range(0, len(self.layers)-1):
            # Clip gradients
            dW[i] = np.clip(dW[i], -1, 1)
            db[i] = np.clip(db[i], -1, 1)
            
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))

    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i])
        return X

    def calculate_loss(self, X, y):
        y_pred = self.predict(X)
        # Thêm epsilon để tránh log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))
