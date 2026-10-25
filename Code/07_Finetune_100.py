import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ================== 加载数据 ==================
X_train = np.load('X_train_100.npy')
y_train = np.load('y_train_100.npy')
X_val = np.load('X_val_100.npy')
y_val = np.load('y_val_100.npy')

print(f"X_train: {X_train.shape}, 类别数: {len(np.unique(y_train))}")

# ================== 标准化 ==================
scaler_path = 'scaler.pkl' if os.path.exists('scaler.pkl') else 'scaler_100.pkl'
print(f"使用标准化器: {scaler_path}")

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

num_samples, timesteps, num_features = X_train.shape
X_train_scaled = scaler.transform(X_train.reshape(-1, num_features)).reshape(num_samples, timesteps, num_features)
X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape[0], timesteps, num_features)

# ================== 加载预训练模型 ==================
base_model = load_model('sign_language_lstm.keras')
print("预训练模型加载成功")
print(f"原模型输入形状: {base_model.input_shape}")
print(f"原模型输出类别数: {base_model.output_shape[-1]}")

# 检查维度
if base_model.input_shape[-1] != num_features:
    print(f"❌ 维度不匹配！预训练模型输入 {base_model.input_shape[-1]}，当前数据 {num_features}")
    print("请先重新训练一个225维的300词模型，或改用下面的备选方案")
    exit()

# ================== 逐层复制（保留预训练权重） ==================
model = Sequential()

# 复制除最后3层（Dense 64, Dropout, Dense 300）外的所有层
for i, layer in enumerate(base_model.layers):
    if i >= len(base_model.layers) - 3:
        break
    layer.trainable = False   # 冻结权重
    model.add(layer)

# 添加新分类头（指定唯一名称，避免重名冲突）
model.add(Dense(64, activation='relu', name='finetune_dense'))
model.add(Dropout(0.3, name='finetune_dropout'))
model.add(Dense(len(np.unique(y_train)), activation='softmax', name='finetune_output'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================== 训练 ==================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=80,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save('sign_language_lstm_100_finetuned.keras')
print("微调模型已保存")

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
plt.savefig('training_curves_100_finetuned.png')
print("训练曲线已保存")