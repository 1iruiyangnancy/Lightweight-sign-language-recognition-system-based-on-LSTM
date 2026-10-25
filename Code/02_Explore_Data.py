import numpy as np
import pickle
from collections import Counter

# 加载数据
X = np.load('X_data.npy')
with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)

print("=== 基本统计 ===")
print(f"样本总数: {len(X)}")
print(f"每个样本的帧数: {X.shape[1]}")
print(f"每帧的特征维度: {X.shape[2]}")
print(f"唯一标签数: {len(set(labels))}")
print(f"数据类型: {X.dtype}")

print("\n=== 标签分布 ===")
label_counts = Counter(labels)
print(f"样本最多的前10个单词:")
for word, count in label_counts.most_common(10):
    print(f"  {word}: {count} 个样本")
print(f"\n样本最少的前10个单词:")
for word, count in label_counts.most_common()[-10:]:
    print(f"  {word}: {count} 个样本")

print(f"\n每个类别平均样本数: {len(labels) / len(set(labels)):.1f}")
print(f"样本最少的类别: {min(label_counts.values())} 个")
print(f"样本最多的类别: {max(label_counts.values())} 个")

print("\n=== 标签示例（前20个） ===")
print(labels[:20])

# 检查是否有任何异常值
print("\n=== 数据质量检查 ===")
print(f"是否有 NaN: {np.isnan(X).any()}")
print(f"是否有 Inf: {np.isinf(X).any()}")
print(f"X 的最小值: {X.min():.4f}")
print(f"X 的最大值: {X.max():.4f}")