# TensorRT-Inject-Preprocess-Postprocess-Layer
In traditional inference pipelines, data is often transferred between CPU and GPU for preprocessing and postprocessing operations. These memory transfers introduce additional latency and synchronization overhead, which can significantly reduce overall inference performance. To address this issue, this project integrates preprocessing and postprocessing directly into the TensorRT execution graph using custom plugin layers backed by CUDA kernels. By keeping the entire pipeline on the GPU, the system minimizes unnecessary memory transfer and enables fully end-to-end GPU inference pipeline.
## How to use
1. Python convert from pytorch model to onnx mode
2. C++ load onnx model then inject preprocess and postprocess layer to onnx layer
3. C++ build engine from onnx
4. C++ inference with engine
## C++ Set up
1. C++ 17
2. MSVC v142
3. CMake >= 3.18
4. Cuda 11.8
5. Cudnn 8.9.7.29
6. TensorRT 8.6.1.6
## Python set up
1. Python 3.8.10
2. Torch 2.0.1+cu118
3. Onnx 1.16.1
## Model for testing
Efficientnet B1
## DEMO
<img width="1623" height="377" alt="image" src="https://github.com/user-attachments/assets/fe235495-89a7-4efe-a83b-b4cfb24bac42" />

