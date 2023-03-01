#ifndef __CUDAI__
#define __CUDAI__

// cudAI
#include "matrix.cuh"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cudai_add(double *, double *, int);
__global__ void cudai_sub(double *, double *, int);
__global__ void cudai_hadamard(double *, double *, int);
__global__ void cudai_dot(double *, double *, int, double *);

#endif