import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# 加载数据
X = np.load('X_data.npy')
with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)

# 统计每个单词的样本数，选最多的100个
counter = Counter(labels)
top100_words = [word for word, _ in counter.most_common(100)]

# 筛选数据
indices = [i for i, label in enumerate(labels) if label in top100_words]
X_100 = X[indices]
labels_100 = [labels[i] for i in indices]

print(f"筛选后: {X_100.shape}, 类别数: {len(set(labels_100))}")
print(f"最少样本数: {min(Counter(labels_100).values())}")
print(f"最多样本数: {max(Counter(labels_100).values())}")

# 重新编码标签
le = LabelEncoder()
y_100 = le.fit_transform(labels_100)

# 保存
np.save('X_data_100.npy', X_100)
np.save('y_data_100.npy', y_100)
with open('label_encoder_100.pkl', 'wb') as f:
    pickle.dump(le, f)
print("已保存 100 词数据")