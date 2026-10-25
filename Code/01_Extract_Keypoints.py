import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
from tqdm import tqdm

print("MediaPipe version:", mp.__version__)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints_from_video(video_path):
    """返回 (frames, 225) 或空数组。225 = 双手(126) + 姿态(99)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0, 225))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = []

        # 左手关键点 (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)

        # 右手关键点 (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)

        # 姿态关键点 (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        frames.append(keypoints)

    cap.release()
    if len(frames) == 0:
        return np.empty((0, 225))
    return np.array(frames, dtype=np.float32)

# 加载标注
print("加载标注文件...")
with open('WLASL300/WLASL_v0.3.json', 'r') as f:
    annotation = json.load(f)

video_to_label = {}
for entry in annotation:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_to_label[video_id] = gloss
print(f"标注条目数: {len(video_to_label)}")

video_dir = 'WLASL300'
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
video_files = sorted(video_files)
print(f"找到 {len(video_files)} 个视频文件")

if len(video_files) == 0:
    exit()

all_sequences = []
all_labels = []
SEQUENCE_LENGTH = 30
skipped_count = 0
skipped_files = []
unknown_count = 0

for video_file in tqdm(video_files, desc="提取关键点"):
    video_path = os.path.join(video_dir, video_file)
    keypoints = extract_keypoints_from_video(video_path)

    if keypoints.shape[0] == 0:
        skipped_count += 1
        skipped_files.append(video_file)
        continue

    num_frames = keypoints.shape[0]
    if num_frames < SEQUENCE_LENGTH:
        pad = SEQUENCE_LENGTH - num_frames
        keypoints = np.pad(keypoints, ((0, pad), (0, 0)), mode='constant')
    else:
        indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        keypoints = keypoints[indices]

    # 从文件名提取最后一个数字作为 video_id
    video_id = video_file.split('_')[-1].split('.')[0]
    label = video_to_label.get(video_id, 'unknown')

    if label == 'unknown':
        unknown_count += 1

    all_sequences.append(keypoints)
    all_labels.append(label)

if len(all_sequences) == 0:
    print("没有成功提取任何视频的关键点，程序退出。")
    exit()

X = np.array(all_sequences, dtype=np.float32)
np.save('X_data.npy', X)
with open('labels.pickle', 'wb') as f:
    pickle.dump(all_labels, f)

print(f"\n=== 统计 ===")
print(f"成功处理: {len(all_sequences)} 个视频")
print(f"跳过（无法提取关键点）: {skipped_count} 个")
print(f"其中标签为 unknown 的: {unknown_count} 个")
print(f"唯一标签数: {len(set(all_labels))}")
print(f"X_data shape: {X.shape}")
if skipped_count > 0:
    print(f"被跳过的视频样例（前5个）: {skipped_files[:5]}")
if len(set(all_labels)) > 1:
    print(f"标签示例（前10个）: {all_labels[:10]}")
else:
    print("警告：所有标签都是 'unknown'，可能 video_id 不匹配！")