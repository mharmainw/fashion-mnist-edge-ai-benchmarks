# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import onnxruntime
import sys


if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")


BEST_LEARNING_RATE = 0.001
BEST_EPOCHS = 20
BEST_MODEL_PATH = "model.pt"
ONNX_MODEL_PATH = "image_classifier_model.onnx"


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


torch_model = BestNeuralNetwork()
torch_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
torch_model.eval()

# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 1, 28, 28),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)

onnx_program.save(ONNX_MODEL_PATH)

onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
print(f"Input length: {len(onnx_inputs)}")

ort_session = onnxruntime.InferenceSession(
    ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

torch_outputs = torch_model(*example_inputs)

torch.testing.assert_close(torch_outputs, torch.tensor(onnxruntime_outputs))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output shape: {onnxruntime_outputs.shape}")
print(f"Sample output: {onnxruntime_outputs}")
