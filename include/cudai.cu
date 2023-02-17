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