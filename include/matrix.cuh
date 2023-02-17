#ifndef __MATRIX_CUDAI__
#define __MATRIX_CUDAI__

#include "cudai.cuh"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <type_traits>

class Matrix {
    public:
        Matrix(); // CONSTRUCTOR / DESTRUCTOR
        ~Matrix();
        size_t get_rows() const { if (shape.size() > 0) return shape[0]; else return 0; }; // SHAPE
        size_t get_cols() const { if (shape.size() > 1) return shape[1]; else return 0; };
        size_t get_cells() const { return cells; }
        void read_file(char *, char delimiter=' ');
        friend std::ostream& operator<<(std::ostream&, const Matrix&); // OPERATORS
        std::vector<double> operator[](long long) const;
        void operator+= (const Matrix &);
        void operator-= (const Matrix &);
        Matrix operator+ (const Matrix &);
        Matrix operator- (const Matrix &);
        double* c_flatten() const; // MANIPULATION
        void hstack(std::vector<double> const&);
        void hstack(std::vector<int> const&);
        void hstack(Matrix);
    private:
        std::vector<std::vector<double>> matrix; // MEMBER VARIABLES
        std::vector<size_t> shape;
        size_t dim;
        size_t cells;
        bool device;
        void read_dim(std::string &);   // IMPORT
        void file_data(std::ifstream &, char);
        void load_values(double *);
        void copy(const Matrix &);
        void rows_inc() { if (get_rows() > 0) shape[0]++; } // SHAPE MANIPULATION
        void cols_inc() { if (get_cols() > 0) shape[1]++; }
        void rows_dec() { if (get_rows() > 0) shape[0]--; } // SHAPE MANIPULATION
        void cols_dec() { if (get_cols() > 0) shape[1]--; }
        void matadd(const Matrix &B); // OPERATIONS
        void matsub(const Matrix &B);
        void cuda_matadd(const Matrix &B); // CUDA
        void cuda_matsub(const Matrix &B);
};
#endif