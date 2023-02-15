#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

class Matrix {
    public:
        Matrix();
        ~Matrix();
        void read_file(char *);
        int get_rows() { return shape[0]; };
        int get_cols() { return shape[1]; };
        double* operator[](int);
        void operator+ (Matrix &);
    private:
        Matrix(Matrix &);
        void read_dim(std::string &);
        void fill_values(std::ifstream &);
        int dim;
        int *shape;
        double **matrix;
};