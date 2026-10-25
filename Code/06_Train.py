import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# ================== 加载数据 ==================
X_train = np.load('X_train_100.npy')
y_train = np.load('y_train_100.npy')
X_val = np.load('X_val_100.npy')
y_val = np.load('y_val_100.npy')

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"类别数: {len(np.unique(y_train))}")

# ================== 特征标准化 ==================
num_samples, timesteps, num_features = X_train.shape

X_train_flat = X_train.reshape(-1, num_features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(num_samples, timesteps, num_features)

X_val_flat = X_val.reshape(-1, num_features)
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], timesteps, num_features)

with open('scaler_100.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("标准化器已保存为 scaler_100.pkl")

# ================== 数据增强 ==================
def augment_sequence(X, y, factor=1):
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(factor):
        X_aug = X.copy()
        for i in range(len(X_aug)):
            shift = np.random.randint(-2, 3)
            if shift != 0:
                X_aug[i] = np.roll(X_aug[i], shift, axis=0)
        noise = np.random.normal(0, 0.01, X_aug.shape)
        X_aug = X_aug + noise
        augmented_X.append(X_aug)
        augmented_y.append(y)
    return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)

X_train_aug, y_train_aug = augment_sequence(X_train_scaled, y_train, factor=1)
print(f"增强后训练集: {X_train_aug.shape}")

# ================== 构建模型 ==================
num_classes = len(np.unique(y_train))

model = Sequential([
    Input(shape=(timesteps, num_features)),

    # 第一层双向 LSTM
    Bidirectional(LSTM(64, return_sequences=True,
                       kernel_regularizer=l2(0.0005),
                       recurrent_regularizer=l2(0.0005))),
    BatchNormalization(),
    Dropout(0.3),

    # 第二层双向 LSTM
    Bidirectional(LSTM(32, return_sequences=False,
                       kernel_regularizer=l2(0.0005),
                       recurrent_regularizer=l2(0.0005))),
    BatchNormalization(),
    Dropout(0.3),

    # 全连接层
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.3),

    # 输出层（自动适配100类）
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================== 回调 ==================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
]

# ================== 训练 ==================
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ================== 保存 ==================
model.save('sign_language_lstm_100.keras')
print("模型已保存为 sign_language_lstm_100.keras")

# ================== 绘制曲线 ==================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title('Loss')
plt.savefig('training_curves_100.png')
print("训练曲线已保存为 training_curves_100.png")