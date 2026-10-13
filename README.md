# OpenVINO Benchmark Reproduction

This repository provides environment validation and baseline inference benchmarks for the reproduction checks described in **Appendix A**.

The benchmark builds an in-memory convolutional model using OpenVINO opset13, feeds a synthetic float32 tensor `(1, 3, 224, 224)`, and measures inference latency across available devices (`CPU` and `GPU`).

---

## 1. System Requirements

* **Python:** 3.9 – 3.12
* **Operating System:** Ubuntu 24.04.5 LTS on WSL 2 or Windows 10/11
* **Hardware Drivers:**
  * **Intel CPU:** Modern x86_64 CPU (AVX2/AVX-512 supported).
  * **Intel GPU (Integrated / Arc dGPU):** [Intel Compute Runtime (NEO)](https://github.com/intel/compute-runtime) and OpenCL drivers.

---

## 2. Environment Setup

Clone this repository and set up a clean Python virtual environment:

```bash
# Clone repository
git clone [https://github.com/Hasnainhai/openvino-benchmark.git](https://github.com/Hasnainhai/openvino-benchmark.git)
cd openvino-benchmark

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
