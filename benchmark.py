import os
import time
import gzip
import struct
from pathlib import Path

import numpy as np


NUM_WARMUP_RUNS = 10
NUM_BENCHMARK_RUNS = 1000
FASHION_MNIST_DIR = Path.home() / ".keras" / "datasets" / "fashion-mnist"


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count, rows, cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number: {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_fashion_mnist_test_data():
    images_path = FASHION_MNIST_DIR / "t10k-images-idx3-ubyte.gz"
    labels_path = FASHION_MNIST_DIR / "t10k-labels-idx1-ubyte.gz"

    images = read_idx_images(images_path).astype("float32") / 255.0
    labels = read_idx_labels(labels_path)
    return images, labels


def summarize_results(name, latencies, accuracy=None):
    mean = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)

    print(f"{name} mean latency: {mean:.4f} ms")
    print(f"{name} P50 latency: {p50:.4f} ms")
    print(f"{name} P99 latency: {p99:.4f} ms")
    if accuracy is not None:
        print(f"{name} accuracy: {accuracy * 100:.2f}%")
    print()


def benchmark_pytorch():
    import torch
    import torch.nn as nn

    class BestNeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = BestNeuralNetwork()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    X_test, y_test = load_fashion_mnist_test_data()
    example_input = torch.randn(1, 1, 28, 28)
    latencies = []
    correct = 0

    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            model(example_input)

        for _ in range(NUM_BENCHMARK_RUNS):
            start = time.perf_counter()
            model(example_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        for image, label in zip(X_test, y_test):
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            prediction = model(image_tensor).argmax(1).item()
            correct += int(prediction == label)

    accuracy = correct / len(y_test)
    summarize_results("PyTorch", latencies, accuracy)


def benchmark_onnx():
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession("image_classifier_model.onnx")

    input_name = session.get_inputs()[0].name
    print(f"ONNX input name: {input_name}")

    X_test, y_test = load_fashion_mnist_test_data()
    example_inputs = (torch.randn(1, 1, 28, 28),)
    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]

    onnxruntime_input = {
        input_arg.name: input_value
        for input_arg, input_value in zip(session.get_inputs(), onnx_inputs)
    }

    latencies = []

    for _ in range(NUM_WARMUP_RUNS):
        session.run(None, onnxruntime_input)

    for _ in range(NUM_BENCHMARK_RUNS):
        start = time.perf_counter()
        session.run(None, onnxruntime_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    correct = 0
    for image, label in zip(X_test, y_test):
        image_input = image.reshape(1, 1, 28, 28).astype("float32")
        prediction = session.run(None, {input_name: image_input})[0].argmax(axis=1)[0]
        correct += int(prediction == label)

    accuracy = correct / len(y_test)
    summarize_results("ONNX Runtime", latencies, accuracy)


def benchmark_keras():
    import tensorflow as tf  # type: ignore

    model = tf.keras.models.load_model("model.keras")
    X_test, y_test = load_fashion_mnist_test_data()
    example_input = np.random.rand(1, 28, 28).astype("float32")
    latencies = []

    for _ in range(NUM_WARMUP_RUNS):
        model.predict(example_input, verbose=0)

    for _ in range(NUM_BENCHMARK_RUNS):
        start = time.perf_counter()
        model.predict(example_input, verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    summarize_results("Keras", latencies, accuracy)


def benchmark_tflite():
    import tensorflow as tf  # type: ignore

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X_test, y_test = load_fashion_mnist_test_data()
    image = np.random.rand(1, 28, 28).astype("float32")

    input_scale, input_zero_point = input_details[0]["quantization"]
    image = image / input_scale + input_zero_point
    image = np.clip(image, -128, 127).astype("int8")

    latencies = []

    for _ in range(NUM_WARMUP_RUNS):
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])

    for _ in range(NUM_BENCHMARK_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    correct = 0
    for image, label in zip(X_test, y_test):
        image = image.reshape(1, 28, 28)
        image = image / input_scale + input_zero_point
        image = np.clip(image, -128, 127).astype("int8")
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"]).argmax(axis=1)[0]
        correct += int(prediction == label)

    accuracy = correct / len(y_test)
    summarize_results("LiteRT/TFLite", latencies, accuracy)


def print_model_sizes():
    model_paths = [
        "model.pt",
        "image_classifier_model.onnx",
        "model.keras",
        "model.tflite",
    ]

    print("Model file sizes:")
    for path in model_paths:
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024
        print(f"{path}: {size_kb:.2f} KB")
    print()


def run_section(name, benchmark_func):
    try:
        benchmark_func()
    except ModuleNotFoundError as error:
        print(f"{name} skipped: missing package {error.name}")
        print()


print_model_sizes()
run_section("PyTorch", benchmark_pytorch)
run_section("ONNX Runtime", benchmark_onnx)
run_section("Keras", benchmark_keras)
run_section("LiteRT/TFLite", benchmark_tflite)
