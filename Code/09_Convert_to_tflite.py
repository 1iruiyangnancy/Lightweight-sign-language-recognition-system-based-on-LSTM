import tensorflow as tf

model = tf.keras.models.load_model('sign_language_lstm_100_finetuned.keras')
print("模型加载成功")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('sign_language_model_100.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite 模型已保存，大小: {len(tflite_model) / 1024:.1f} KB")