#include "cuchine.cuh"

__global__
void cuchine_matadd(double *a, double *b, int entries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < entries) {
        a[idx] += b[idx];
    }
}

__global__
void cuchine_matsub(double *a, double *b, int entries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < entries) {
        a[idx] -= b[idx];
    }
}