import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_model(timesteps, num_features, num_classes):
    """简化版模型：单个双向 LSTM + 轻量正则化"""
    model = Sequential([
        Input(shape=(timesteps, num_features)),

        # 只用一个双向 LSTM 层，大幅减少参数
        Bidirectional(LSTM(32, return_sequences=False,
                           kernel_regularizer=l2(0.0003),
                           recurrent_regularizer=l2(0.0003))),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(0.0003)),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model