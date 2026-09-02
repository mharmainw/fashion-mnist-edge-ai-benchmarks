import numpy as np
import tensorflow as tf #type: ignore

_ , (X_test,_) = tf.keras.datasets.fashion_mnist.load_data()


interpreter = tf.lite.Interpreter(
    model_path = 'model.tflite'
)



interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

image = X_test[0:1].astype("float32") / 255.0
input_scale, input_zero_point = input_details[0]["quantization"]
image = image / input_scale + input_zero_point
image = np.clip(image, -128, 127).astype("int8")

interpreter.set_tensor(
    input_details[0]["index"],
    image
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

output_scale, output_zero_point = output_details[0]["quantization"]
output = (output.astype("float32") - output_zero_point) * output_scale

prediction = output.argmax(axis=1)
print(f"Prediction: {prediction[0]}")
print(f"Scores: {output[0]}")
