import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# 加载测试数据
X_test = np.load('X_test_100.npy')
y_test = np.load('y_test_100.npy')

# 加载模型
model = load_model('sign_language_lstm_100_finetuned.keras')

# 加载标准化器
with open('scaler_100.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 标准化测试集
num_samples, timesteps, num_features = X_test.shape
X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(num_samples, timesteps, num_features)

# 预测
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# 准确率
acc = np.mean(y_pred == y_test)
print(f"测试集准确率: {acc:.4f}")

# 加载标签编码器
with open('label_encoder_100.pkl', 'rb') as f:
    le = pickle.load(f)
class_names = le.classes_
print(f"类别总数: {len(class_names)}")

# 分类报告
print("\n分类报告（前20类）:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# 混淆矩阵（前20类）
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm[:20, :20], annot=True, fmt='d',
            xticklabels=class_names[:20],
            yticklabels=class_names[:20])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (first 20 classes) - Acc: {acc:.2%}')
plt.tight_layout()
plt.savefig('confusion_matrix_100.png')
print("混淆矩阵已保存")