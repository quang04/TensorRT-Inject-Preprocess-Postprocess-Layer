#include <cuda_runtime.h>
#include <math_constants.h>
#include "tensorRTKernel.h"

#define K 5
#define NUM_THREAD 256

__device__ void createTopK(float value, int indexValue, float* listVal, float* listIndex) 
{
    for(int i=0; i<K; i++)
    {
        if(value > listVal[i]) 
        {
            // shift down
            for(int j = K-1; j>i; j--) {
                listVal[j] = listVal[j-1];
                listIndex[j] = listIndex[j-1];
            }

            // append new value at current position
            listVal[i] = value;
            listIndex[i] = indexValue;
            break;
        }
    }

}

__global__ void topkSoftmax(const float* __restrict__ input, float* __restrict__ output, int n)
{
    __shared__ float shareVal[NUM_THREAD][K];
    __shared__ int shareIndex[NUM_THREAD][K];

    int tIdx = threadIdx.x;

    // compute local top k for each thread
    float localVal[K];
    float localIndex[K];

    #pragma unroll K
    for(int i=0; i<K; i++) 
    {
        localVal[i] = -CUDART_INF_F;
        localIndex[i] = -1;
    }

    for(int i=tIdx; i<n; i+=NUM_THREAD)
    {
        float v = input[i];
        createTopK(v, i, localVal, localIndex);
    }

    // stored to share value
    #pragma unroll K
    for(int i=0; i<K; i++) 
    {
        shareVal[tIdx][i] = localVal[i];
        shareIndex[tIdx][i] = localIndex[i];
    }

    // make sure all thread complete
    __syncthreads();

    // combine to global top k
    if(tIdx == 0) 
    {
        float finalVal[K];
        float finalIndex[K];

        #pragma unroll K
        for (int i = 0; i < K; i++) {
            finalVal[i] = -CUDART_INF_F;
            finalIndex[i] = -1;
        }

        // merge
        for(int i=0; i<NUM_THREAD; i++)
        {
            #pragma unroll K
            for(int j=0; j<K; j++)
            {
                createTopK(shareVal[i][j], shareIndex[i][j], finalVal, finalIndex);
            }
        }

        // calculate softmax
        float maxVal = finalVal[0];

        float sum = 0.0f;
        float score[K];

        #pragma unroll K
        for(int i=0; i<K; i++)
        {
            float e = __expf(finalVal[i] - maxVal);
            score[i] = e;
            sum += e;
        }

        float inv = 1.0f/sum;
        #pragma unroll K
        for(int i=0; i<K; i++)
        {
            score[i] *= inv;
        }

        // write to result
        #pragma unroll K
        for(int i=0; i<K; i++)
        {
            output[i] = finalIndex[i];
            output[i+K] = score[i];
        }

    }
}

__device__ float lerp(float a, float b, float t)
{
    return a + (b-a)*t;
}

__global__ void preprocess(const unsigned char* __restrict__ in, float* __restrict__ out,
    int inH, int inW, int outH, int outW,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outH * outW;
    if(idx >= total) return;

    // (x,y) row major
    int x = idx / outW;
    int y = idx % outW;

    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    // mapback to input pixel position
    float xSrc = (x + 0.5f)*scaleX - 0.5f;
    float ySrc = (y + 0.5f)*scaleY - 0.5f;

    // find the neigbor of pixel
    int x0 = floorf(xSrc);
    int y0 = floorf(ySrc);
    int x1 = min(x0 + 1, inW - 1);
    int y1 = min(y0 + 1, inH - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    // find the neigbor of pixel RGB
    int i00 = (y0 + x0*inW)*3;
    int i01 = (y0 + x1*inW)*3;
    int i10 = (y1 + x0*inW)*3;
    int i11 = (y1 + x1*inW)*3;

    float r00 = in[i00 + 0];
    float g00 = in[i00 + 1];
    float b00 = in[i00 + 2];

    float r01 = in[i01 + 0];
    float g01 = in[i01 + 1];
    float b01 = in[i01 + 2];

    float r10 = in[i10 + 0];
    float g10 = in[i10 + 1];
    float b10 = in[i10 + 2];

    float r11 = in[i11 + 0];
    float g11 = in[i11 + 1];
    float b11 = in[i11 + 2];

    float lx = xSrc - x0;
    float ly = ySrc - y0;

    // bilinear interpolate
    float r0 = lerp(r00, r01, lx);
    float r1 = lerp(r10, r11, lx);
    float r  = lerp(r0,  r1,  ly);

    float g0 = lerp(g00, g01, lx);
    float g1 = lerp(g10, g11, lx);
    float g  = lerp(g0,  g1,  ly);
        
    float b0 = lerp(b00, b01, lx);
    float b1 = lerp(b10, b11, lx);
    float b  = lerp(b0,  b1,  ly);

    // normalize
    r = (r / 255.0f - mean0) / std0;
    g = (g / 255.0f - mean1) / std1;
    b = (b / 255.0f - mean2) / std2;

    // write output as CHW format (r1r2r3..rN g1g2g3..gN b1b2b3..bN)
    int pixelIndex = x*outW + y;
    out[total*0 + pixelIndex] = r;
    out[total*1 + pixelIndex] = g;
    out[total*2 + pixelIndex] = b;

}

extern "C" void launchTopKSoftmaxKernel(float* out, const float* in, int n, cudaStream_t stream)
{
    // calculate thread local top k to take advantage of loop unrolling
    topkSoftmax<<<1, NUM_THREAD, 0, stream>>>(in, out, n);
}

extern "C" void launchPreprocessKernel(const unsigned char *in, float *out, int inH, int intW, int outH, int outW, float mean0, float mean1, float mean2, float std0, float std1, float std2, cudaStream_t stream)
{
    int total = outW*outH;
    int numBlock = (total + NUM_THREAD - 1) / NUM_THREAD;

    preprocess<<<numBlock, NUM_THREAD, 0, stream>>>(in, out, inH, intW, outH, outW, mean0, mean1, mean2, std0, std1, std2);
}