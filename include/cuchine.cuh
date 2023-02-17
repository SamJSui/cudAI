#ifndef __CUCHINE__
#define __CUCHINE__

// CUchine
#include "matrix.cuh"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuchine_matadd(double *, double *, int);
__global__ void cuchine_matsub(double *, double *, int);

#endif