# Week 7 - TinyML & Edge AI

This repository contains the Week 7 TinyML / Edge AI work for Fashion-MNIST model export, quantisation, and inference benchmarking.

## Files

| File | Purpose |
| --- | --- |
| `best_model_setup.py` | Loads the Week 3 PyTorch model and exports it to ONNX. |
| `benchmark.py` | Benchmarks PyTorch, ONNX Runtime, and LiteRT/TFLite when the required packages are available. |
| `converter.py` | Converts `model.keras` to an INT8 quantised TFLite model. |
| `keras_model.py` | Trains the Keras Fashion-MNIST model and saves `model.keras`. |
| `tflite_model.py` | Runs inference with the quantised TFLite model. |
| `model.pt` | Saved PyTorch model weights. |
| `image_classifier_model.onnx` | ONNX export of the PyTorch model. |
| `model.keras` | Saved Keras model before TFLite quantisation. |
| `model.tflite` | INT8 quantised LiteRT/TFLite model. |

## Benchmark Results

Latency benchmarks use 10 warmup calls followed by 1,000 timed inference calls.

The results below come from two separate model pipelines:

```text
PyTorch model -> ONNX model
Keras model -> INT8 LiteRT/TFLite model
```

The PyTorch/ONNX results should be compared with each other because they use the same source model. The Keras/TFLite results should be compared with each other because `model.tflite` is the quantised version of `model.keras`.

The TFLite accuracy should not be interpreted as being higher than the PyTorch model accuracy, because the TFLite model comes from a different Keras model.

### PyTorch to ONNX

| Format | Model file | Size | Mean latency | P50 latency | P99 latency | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | `model.pt` | 2.56 MB | 0.1093 ms | 0.1042 ms | 0.2641 ms | 88.09% |
| ONNX Runtime | `image_classifier_model.onnx` | 2.56 MB | 0.0475 ms | 0.0396 ms | 0.1203 ms | 88.09% |

### Keras to LiteRT/TFLite

| Format | Model file | Size | Mean latency | P50 latency | P99 latency | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Keras | `model.keras` | 1.19 MB | 52.9581 ms | 49.8232 ms | 88.0902 ms | 89.30% |
| LiteRT/TFLite INT8 | `model.tflite` | 0.10 MB | 0.0043 ms | 0.0043 ms | 0.0048 ms | 89.34% |

Accuracy drop after INT8 quantisation:

```text
Keras accuracy - TFLite accuracy = 89.30% - 89.34% = -0.04 percentage points
```

In this run, the quantised TFLite model did not show an accuracy drop.

## Notes

The PyTorch/ONNX model and the Keras/TFLite model are not the exact same architecture, so the benchmark demonstrates the Week 7 export and quantisation workflow rather than a perfectly controlled runtime comparison across all four formats.

The local environments are split:

```text
.venv     -> PyTorch and ONNX Runtime
.venv311  -> TensorFlow and LiteRT/TFLite
```

Because of this, `benchmark.py` skips benchmark sections when a required package is missing from the active environment.
