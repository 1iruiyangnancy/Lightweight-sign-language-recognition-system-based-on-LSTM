import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf

# ================== 加载 TFLite 模型 ==================
print("加载 TFLite 模型...")
interpreter = tf.lite.Interpreter(model_path='sign_language_model_100.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"输入形状: {input_details[0]['shape']}")
print(f"输出形状: {output_details[0]['shape']}")

# 加载标签
with open('label_encoder_100.pkl', 'rb') as f:
    le = pickle.load(f)
class_names = le.classes_
print(f"已加载 {len(class_names)} 个单词")

# 加载标准化器
with open('scaler_100.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ================== 初始化MediaPipe ==================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ================== 初始化摄像头 ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225
sequence = []
confidence_threshold = 0.5


def extract_keypoints(results):
    """提取225维特征"""
    keypoints = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 63)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 99)
    return np.array(keypoints, dtype=np.float32)


def predict_tflite(input_data):
    """用 TFLite 模型推理"""
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]


print("开始识别... 按 'q' 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]

    if len(sequence) == SEQUENCE_LENGTH:
        seq_array = np.array(sequence, dtype=np.float32)  # (30, 225)

        # 标准化
        seq_flat = seq_array.reshape(-1, FEATURE_DIM)
        seq_scaled = scaler.transform(seq_flat).reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)
        seq_scaled = seq_scaled.astype(np.float32)

        # TFLite 推理
        pred_probs = predict_tflite(seq_scaled)
        pred_class = np.argmax(pred_probs)
        confidence = np.max(pred_probs)

        if confidence > confidence_threshold:
            top3_idx = np.argsort(pred_probs)[-3:][::-1]
            y_offset = 40
            for i, idx in enumerate(top3_idx):
                label = class_names[idx]
                prob = pred_probs[idx]
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(frame, f"{i+1}. {label}: {prob:.2f}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7 if i == 0 else 0.5, color, 2)
                y_offset += 30

    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()