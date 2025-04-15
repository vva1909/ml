import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
# Tạo 1000 điểm dữ liệu ngẫu nhiên
np.random.seed(42)
X = np.random.rand(1000, 2)  # 1000 điểm với 2 đặc trưng
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Nhãn nhị phân

# Chia dữ liệu thành tập train và test
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Tạo mô hình Neural Network
model = keras.Sequential([
    # Lớp đầu vào với 2 đặc trưng
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    # Lớp ẩn
    keras.layers.Dense(4, activation='relu'),
    # Lớp đầu ra với hàm sigmoid cho phân loại nhị phân
    keras.layers.Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Vẽ đồ thị quá trình huấn luyện
plt.figure(figsize=(12, 4))

# Vẽ accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Vẽ loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Dự đoán trên dữ liệu mới
new_data = np.array([[0.2, 0.3], [0.8, 0.9]])
predictions = model.predict(new_data)
print("\nPredictions for new data:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred[0]:.4f} (Class: {1 if pred[0] > 0.5 else 0})") 