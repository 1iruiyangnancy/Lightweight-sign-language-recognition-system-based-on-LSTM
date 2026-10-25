import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# 加载100词数据
X = np.load('X_data_100.npy')
y = np.load('y_data_100.npy')   # 已经是数字标签

print(f"原始数据: {X.shape}, 类别数: {len(np.unique(y))}")

# 划分训练/验证/测试集 (70% / 15% / 15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")
print(f"测试集: {X_test.shape}")

# 保存（使用 _100 后缀，避免覆盖之前300词的数据）
np.save('X_train_100.npy', X_train)
np.save('X_val_100.npy', X_val)
np.save('X_test_100.npy', X_test)
np.save('y_train_100.npy', y_train)
np.save('y_val_100.npy', y_val)
np.save('y_test_100.npy', y_test)

print("数据已保存！")