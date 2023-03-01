#include "cudai.cuh"

__global__
void cudai_add(double *a, double *b, int cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cells) {
        a[idx] += b[idx];
    }
}

__global__
void cudai_sub(double *a, double *b, int cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cells) {
        a[idx] -= b[idx];
    }
}

__global__
void cudai_hadamard(double *a, double *b, int cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cells) {
        a[idx] *= b[idx];
    }
}

__global__ void cudai_dot(double *a, double *b, int cells, double *scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cells) {
        *scalar += a[idx] * b[idx];
    }
}