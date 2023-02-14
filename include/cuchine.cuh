// CUchine

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
// I/O
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// DATA STRUCTURES
#include <strings.h>
#include <string>
#include <vector>

class Matrix {
    public:
        Matrix();
        ~Matrix();
        void read_file(char *);
        int get_rows() { return shape[0]; };
        int get_cols() { return shape[1]; };
        std::vector<double> operator[](int);
    private:
        void read_dim(std::string &);
        int dim;
        int *shape;
        std::vector< std::vector<double> > matrix;
};