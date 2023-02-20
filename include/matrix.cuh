#ifndef __MATRIX_CUDAI__
#define __MATRIX_CUDAI__

#include "cudai.cuh"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>

class Matrix {
    public:
        Matrix(); // CONSTRUCTOR / DESTRUCTOR
        ~Matrix();
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
        // MANIPULATION
        double* c_flatten() const;
        void hstack(std::vector<double> const&);
        void hstack(std::vector<int> const&);
        void hstack(Matrix);
    private:
        // MEMBER VARIABLES
        std::vector<std::vector<double>> matrix;
        std::vector<size_t> shape;
        size_t cells;
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
        void hadamard(const Matrix &B);
        void dotprod(const Matrix &B);
        // CUDA
        void cuda_matadd(const Matrix &B);
        void cuda_matsub(const Matrix &B);
        void cuda_hadamard(const Matrix &B);
        void cuda_dotprod(const Matrix &B);
};
#endif