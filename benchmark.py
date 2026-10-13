
import time
import numpy as np
import openvino as ov

core = ov.Core()

# Small synthetic input for an initial runtime test.
# We'll replace this with a real ONNX model next.
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Simple convolutional model to test device execution.
import openvino as ov
from openvino import opset13 as ops

data = ops.parameter([1, 3, 224, 224], np.float32, name="input")
weights = ops.constant(
    np.random.rand(8, 3, 3, 3).astype(np.float32)
)
conv = ops.convolution(
    data, weights, strides=[1, 1],
    pads_begin=[1, 1], pads_end=[1, 1],
    dilations=[1, 1]
)
model = ov.Model([conv], [data], "baseline_conv")

for device in ["CPU", "GPU"]:
    print(f"\nTesting {device}...")
    compiled = core.compile_model(model, device)
    request = compiled.create_infer_request()

    # Warm-up runs; excluded from the timing.
    for _ in range(10):
        request.infer({0: input_data})

    times_ms = []
    for _ in range(100):
        start = time.perf_counter()
        request.infer({0: input_data})
        times_ms.append((time.perf_counter() - start) * 1000)

    print(f"Mean latency:   {np.mean(times_ms):.3f} ms")
    print(f"Median latency: {np.median(times_ms):.3f} ms")
    print(f"Maximum observed: {np.max(times_ms):.3f} ms")
