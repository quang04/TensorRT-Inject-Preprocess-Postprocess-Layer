#pragma once

#include <cuda_runtime.h>

extern "C" void launchTopKSoftmaxKernel(float* out, const float* in, int n, cudaStream_t stream);

extern "C" void launchPreprocessKernel(const unsigned char* in, float* out, int inH, int intW, int outH, int ouW,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2,
    cudaStream_t stream);