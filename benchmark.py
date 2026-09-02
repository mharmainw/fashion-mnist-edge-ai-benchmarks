import os
import time

import numpy as np


NUM_WARMUP_RUNS = 10
NUM_BENCHMARK_RUNS = 1000


def summarize_latencies(name, latencies):
    mean = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)

    print(f"{name} mean latency: {mean:.4f} ms")
    print(f"{name} P50 latency: {p50:.4f} ms")
    print(f"{name} P99 latency: {p99:.4f} ms")
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

    example_input = torch.randn(1, 1, 28, 28)
    latencies = []

    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            model(example_input)

        for _ in range(NUM_BENCHMARK_RUNS):
            start = time.perf_counter()
            model(example_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    summarize_latencies("PyTorch", latencies)


def benchmark_onnx():
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession("image_classifier_model.onnx")

    input_name = session.get_inputs()[0].name
    print(f"ONNX input name: {input_name}")

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

    summarize_latencies("ONNX Runtime", latencies)


def benchmark_tflite():
    import tensorflow as tf  # type: ignore

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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

    summarize_latencies("LiteRT/TFLite", latencies)


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
run_section("LiteRT/TFLite", benchmark_tflite)
