import numpy as np

def my_train_test_split(X, y, test_size=0.2, random_state=None):
    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X va Y khong cung chieu du lieu")

    n_samples = X.shape[0]

    n_train = int(n_samples - (n_samples * test_size))

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test