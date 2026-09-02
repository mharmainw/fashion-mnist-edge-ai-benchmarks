import tensorflow as tf # type: ignore


(X_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255.0

model = tf.keras.models.load_model("model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_data_gen():
    for i in range(100):
        yield [X_train[i:i+1].astype("float32")]

converter.optimizations  = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()



with open("model.tflite", "wb") as f:
    f.write(tflite_model)
