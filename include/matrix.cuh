#ifndef __MATRIX_CUCHINE__
#define __MATRIX_CUCHINE__

#include "cuchine.cuh"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>

class Matrix {
    public:
        Matrix(); // CONSTRUCTOR / DESTRUCTOR
        ~Matrix();
        long long get_rows() const { if (shape.size() > 0) return shape[0]; else return 0; }; // SHAPE
        long long get_cols() const { if (shape.size() > 1) return shape[1]; else return 0; };
        void read_file(char *, char delimiter=' ');
        void copy(const Matrix &);
        friend std::ostream& operator<<(std::ostream&, const Matrix&); // OPERATORS
        std::vector<double> operator[](long long) const;
        void operator+= (const Matrix &);
        void operator-= (const Matrix &);
        Matrix operator+ (const Matrix &);
        Matrix operator- (const Matrix &);
        double* c_flatten() const;
    private:
        std::vector<std::vector<double>> matrix; // MEMBER VARIABLES
        std::vector<size_t> shape;
        size_t dim;
        size_t entries;
        bool device;
        void read_dim(std::string &); // IMPORT
        void fill_values(std::ifstream &, char);
        void load_values(double *);
        void matadd(const Matrix &B);
        void matsub(const Matrix &B);
        void cuda_matadd(const Matrix &B); // CUDA
        void cuda_matsub(const Matrix &B);
};
#endif