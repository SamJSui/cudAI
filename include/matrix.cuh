#ifndef __MATRIX_CUDAI__
#define __MATRIX_CUDAI__

#include "cudai.cuh"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>

class Matrix {
    public:
        // CONSTRUCTOR / DESTRUCTOR
        Matrix();
        Matrix(const size_t, const size_t);
        ~Matrix();
        // MEMBER VARIABLES
        size_t cells, rows, col;
        // SHAPE
        size_t get_rows() const { return shape[0]; };
        size_t get_cols() const { return shape[1]; };
        size_t get_cells() const { return cells; }
        void read_file(char *, char delimiter=' ');
        // OPERATORS
        friend std::ostream& operator<<(std::ostream&, const Matrix&);
        std::vector<double> operator[](long long) const;
        void operator+= (const Matrix &);
        void operator-= (const Matrix &);
        void operator*= (const Matrix &);
        Matrix operator+ (const Matrix &);
        Matrix operator- (const Matrix &);
        Matrix operator* (const Matrix &);
        Matrix add(const Matrix &B); 
        Matrix sub(const Matrix &B);
        Matrix mul(const Matrix &B);
        void hadamard(const Matrix &B);
        // MANIPULATION
        double* c_flatten() const;
        double** c_alloc() const;
        void hstack(std::vector<double> const&);
        void hstack(std::vector<int> const&);
        void hstack(Matrix);
    private:
        // MEMBER VARIABLES
        std::vector<std::vector<double>> matrix;
        std::vector<size_t> shape;
        bool device;
        // IMPORT
        void file_data(std::ifstream &, char);
        void load_values(double *);
        void copy(const Matrix &);
        // SHAPE MANIPULATION
        void rows_inc() { shape[0]++; } 
        void cols_inc() { shape[1]++; }
        void rows_dec() { shape[0]--; }
        void cols_dec() { shape[1]--; }
        // OPERATIONS
        void matadd(const Matrix &B); 
        void matsub(const Matrix &B);
        Matrix matmul(const Matrix &B);
        // CUDA
        void cuda_matadd(const Matrix &B);
        void cuda_matsub(const Matrix &B);
        Matrix cuda_matmul(const Matrix &B);
        void cuda_hadamard(const Matrix &B);
};
#endif