import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Yêu cầu 1: Khởi tạo class LinearRegression
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, bias=True):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = bias

    def predict(self, X):
        """
        Hàm dự đoán doanh số dựa trên chi phí quảng cáo TV, Radio, Newspaper.
        """
        if self.weights is None:
            raise ValueError("Mô hình chưa được huấn luyện.")

        return np.dot(X, self.weights) + self.bias

    def loss_function(self, y_true, y_pred):
        """
        Hàm tính toán MSE (Mean Squared Error) giữa giá trị thực tế và giá trị dự đoán.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def gradient(self, X, y, y_pred):
        """
        Hàm tính gradient của loss function đối với các tham số.
        """
        m = len(y)
        dw = -2 * np.dot(X.T, (y - y_pred)) / m
        db = -2 * np.sum(y - y_pred) / m
        return dw, db

    def fit(self, X, y):
        """
        Hàm huấn luyện mô hình bằng cách cập nhật trọng số (weights) và hệ số chặn (bias).
        """
        m, n = X.shape
        # Khởi tạo trọng số ngẫu nhiên
        self.weights = np.random.randn(n)
        loss_history = []

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            dw, db = self.gradient(X, y, y_pred)

            # Cập nhật trọng số và hệ số chặn
            self.weights -= self.lr * dw
            if self.bias:
                self.bias -= self.lr * db

            # Tính toán loss
            loss = self.loss_function(y, y_pred)
            loss_history.append(loss)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return loss_history


# Yêu cầu 2: Triển khai trên bộ dữ liệu Advertising

# Đọc dữ liệu
df = pd.read_csv('data/advertising.csv')

# Trích xuất đặc trưng TV, Radio, Newspaper và Sales
X = df[['TV', 'Radio', 'Newspaper']].values
y = df['Sales'].values

# Chuẩn hóa dữ liệu (Min-Max Scaling)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression(learning_rate=0.01, epochs=1000)

# Huấn luyện mô hình
loss_history = model.fit(X_train, y_train)

# Vẽ learning curve
plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='blue')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.show()

# Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán các metric đánh giá
mse = model.loss_function(y_test, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

# In kết quả đánh giá
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared: {r2}')

# Trực quan hóa kết quả dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Perfect fit')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.legend()
plt.show()