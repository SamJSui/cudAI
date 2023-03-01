#include "matrix.cuh"

//// PUBLIC

Matrix::Matrix() {
    shape.resize(2, 0);
    cells = 0;
    device = 0;                                    // Device check
    cudaGetDeviceCount((int *) &device);
}

Matrix::~Matrix() {
    ;
}

void Matrix::read_file(char *filename, char delimiter) {
    // READING IN FILE AGAIN
    if (matrix.size() > 0) {
        size_t i;
        for (i = 0; i < get_rows(); i++) { matrix[i].clear(); }
        shape[0] = 0;
        shape[1] = 0;
    }

    // DECLARE
    std::string line, num;
    size_t rows, cols;

    // FILE HANDLING
    std::ifstream ifs(filename, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: File did not open!\n";
        exit(EXIT_FAILURE);
    }

    // INITIALIZE
    rows = 0;
    cols = 0;
    while (getline(ifs, line)) {
        if (cols == 0) {
            std::istringstream iss(line);
            while(getline(iss, num, delimiter)) {
                cols++;
            }
        }
        rows++;
    }

    // MATRIX SHAPE
    shape[0] = rows;
    shape[1] = cols;
    cells = rows * cols;

    // MATRIX VALUES
    ifs.clear();
    ifs.seekg(0, std::ios::beg);
    file_data(ifs, delimiter);

    // CLOSE FILE
    ifs.close();
}

// OPERATORS

std::vector<double> Matrix::operator[](long long idx) const {
    if (get_rows() > 0 && idx > -1 && idx < get_rows()) return matrix[idx];
    else { 
        std::cerr << "ERROR: Invalid index accessing matrix!\n";
        return {};
    }
}

std::ostream& operator<< (std::ostream &os, const Matrix &M) {
    size_t i, j;
    os << "[";
    for (i = 0; i < M.get_rows(); i++) {
        os << "[ ";
        for (j = 0; j < M.get_cols(); j++) {
            os << M[i][j] << ' ';
        }
        os << "]";
        if (i < M.get_rows()-1) os << "\n";
    }
    os << "]";
    return os;
}

void Matrix::operator+= (const Matrix &B) {
    // ERROR CHECK
    if (get_rows() != B.get_rows() || get_cols() != B.get_cols()) {
        fprintf(stderr, "ERROR: Invalid dimensions for '+=' operation!\n");
        exit(1);
    }
    if (device) { cuda_matadd(B); }
    else matadd(B);
}

void Matrix::operator-= (const Matrix &B) {
    // ERROR CHECK
    if (get_rows() != B.get_rows() || get_cols() != B.get_cols()) {
        fprintf(stderr, "ERROR: Invalid dimensions for '-=' operation!\n");
        exit(1);
    }

    if (device) cuda_matsub(B);
    else matsub(B);
}

void Matrix::operator*= (const Matrix &B) {
    if (get_rows() == 1 && B.get_rows() == 1 && get_cols() == B.get_cols()) {
        if (device) { cuda_hadamard(B); }
        else hadamard(B);
    }
}

Matrix Matrix::operator+(const Matrix &B) {
    Matrix C;
    C.copy(*this);
    C += B;
    return C;
}

Matrix Matrix::operator-(const Matrix &B) {
    Matrix C;
    C.copy(*this);
    C -= B;
    return C;
}

Matrix Matrix::operator*(const Matrix &B) {
    if (get_rows() == B.get_rows() && get_cols() == B.get_cols()) {
        Matrix C;
        C.copy(*this);
        C.hadamard(B);
        return C;
    }
    else if (get_cols() == B.get_rows()) {
        Matrix C(get_rows(), B.get_cols());
        C.matmul(B);
        return C;
    }
    else {
        return Matrix();
    }
}

Matrix Matrix::add(const Matrix &B) {
    Matrix C;
    C.copy(*this);
    C += B;
    return C;
}

Matrix Matrix::sub(const Matrix &B) {
    Matrix C;
    C.copy(*this);
    C -= B;
    return C;
}

Matrix Matrix::mul(const Matrix &B) {
    if (get_rows() == B.get_rows() && get_cols() == B.get_cols()) {
        Matrix C;
        C.copy(*this);
        C.hadamard(B);
        return C;
    }
    else if (get_cols() == B.get_rows()) {
        Matrix C(get_rows(), B.get_cols());
        C.matmul(*this, B);
        return C;
    }
    else {
        return Matrix();
    }
}

void Matrix::hadamard(const Matrix &B) {
    if (device) { cuda_hadamard(B); }
    else {
        size_t i, j;
        for (i = 0; i < get_rows(); i++) {
            for (j = 0; j < get_cols(); j++) {
                matrix[i][j] *= B[i][j];
            }
        }
    }
}

// MANIPULATION

double* Matrix::c_flatten() const {
    // DECLARE
    size_t rows, cols, i, j, idx;
    double *tmp;

    // INITIALIZE
    rows = get_rows();
    cols = get_cols();
    tmp = (double *) malloc(cells * sizeof(double));

    // LOAD
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = rows * i + j;
            tmp[idx] = matrix[i][j];
        }
    }
    return tmp;
}

double** Matrix::c_alloc() const {
    // DECLARE
    double **c_Matrix;
    size_t i, j;

    // INITIALIZE
    c_Matrix = (double **) malloc(get_rows() * sizeof(double *));
    for (i = 0; i < get_rows(); i++) {
        c_Matrix[i] = (double *) malloc(get_cols() * sizeof(double));
        for (j = 0; j < get_cols(); j++) {
            c_Matrix[i][j] = matrix[i][j];
        }
    }
    return c_Matrix;
}

void Matrix::hstack(std::vector<double> const &row) {
    if (row.size() == get_cols()) {
        matrix.push_back(row);
        rows_inc();
    }
    else {
        fprintf(stderr, "hstack() requires equal number of columns!\n");
    }
}

void Matrix::hstack(std::vector<int> const &row) {
    if (row.size() == get_cols()) {
        std::vector<double> tmp(row.begin(), row.end());
        matrix.push_back(tmp);
        rows_inc();
    }
    else {
        fprintf(stderr, "hstack() requires equal number of columns!\n");
    }
}

void Matrix::hstack(Matrix B) {
    size_t i;
    if (get_cols() == B.get_cols()) {
        for (i = 0; i < B.get_rows(); i++) {
            matrix.push_back(B[i]);
            rows_inc();
        }
    }
    else {
        fprintf(stderr, "hstack() requires equal number of columns!\n");
    }
}

//// PRIVATE

// CONSTRUCTOR/DESTRUCTOR
Matrix::Matrix(const size_t rows, const size_t cols) {
    size_t i;
    matrix.resize(rows);
    for (i = 0; i < rows; i++) {
        matrix[i].resize(cols);
    }
    shape.resize(2, 0);
    shape[0] = rows;
    shape[1] = cols;
    cells = rows * cols;
    device = 0;                                    // Device check
    cudaGetDeviceCount((int *) &device);
}

// MANIPULATION
void Matrix::file_data(std::ifstream &ifs, char delimiter) {
    // DECLARE
    std::string line, num;
    size_t rows, cols, lineNum, i, j;
    std::istringstream iss;

    // INITIALIZE
    rows = get_rows(), cols = get_cols();
    i = 0, j = 0;
    lineNum = 2;

    while (getline(ifs, line)) {
        iss = std::istringstream(line);
        std::vector<double> tmp_row;
        while (getline(iss, num, delimiter)) {
            if (j == cols) { std::cerr << "ERROR: Line " << lineNum << " has too many values has too many values in the row!\n"; exit(1); }
            try { tmp_row.push_back(stod(num)); }
            catch (std::exception &e) {
                fprintf(stderr, "ERROR: Line %lu has input data has invalid characters within the values!\n %s\n", lineNum, e.what());
                exit(1);
            }
            j++;
        }
        matrix.push_back(tmp_row);
        if (j != cols) {
            fprintf(stderr, "ERROR: Not enough values in input file columns!\n");
            exit(1);
        }
        j = 0;
        i++;
    }
    if (i != rows) {
        fprintf(stderr, "ERROR: Not enough values in input file rows!\n");
        exit(1);
    }
}

void Matrix::load_values(double *d) {
    // DECLARE
    unsigned long long i, j;

    // LOAD
    for (i = 0; i < get_rows(); i++) {
        for (j = 0; j < get_cols(); j++) {
            int idx = rows * i + j;
            matrix[i][j] = d[idx];
        }
    }
}

void Matrix::copy(const Matrix &B){
    // DECLARE
    size_t i, j;
    
    // INITIALIZE
    shape.resize(2);
    shape[0] = B.get_rows();
    shape[1] = B.get_cols();

    // LOAD
    matrix.resize(shape[0], std::vector<double>(shape[1]));
    for (i = 0; i < get_rows(); i++) {
        for (j = 0; j < get_cols(); j++) {
            matrix[i][j] = B[i][j];
        }
    }
}

// OPERATIONS
void Matrix::matadd(const Matrix &B) {
    size_t i, j;
    for (i = 0; i < get_rows(); i++) {
        for (j = 0; j < get_cols(); j++) {
            matrix[i][j] += B[i][j];
        }
    }
}

void Matrix::matsub(const Matrix &B) {
    size_t i, j;
    for (i = 0; i < get_rows(); i++) {
        for (j = 0; j < get_cols(); j++) {
            matrix[i][j] += B[i][j];
        }
    }
}

Matrix Matrix::matmul(const Matrix &A, const Matrix &B) {

}

// CUDA
void Matrix::cuda_matadd(const Matrix &B) {
    // DECLARE
    size_t bytes, N_BLOCKS, N_THREADS;
    double *a_flat, *b_flat, *a_flat_device, *b_flat_device;

    // INITIALIZE
    a_flat = c_flatten();
    b_flat = B.c_flatten();
    bytes = cells * sizeof(double);
    N_THREADS = 1024;
    N_BLOCKS = cells/1024 + 1;

    cudaMalloc((void **) &a_flat_device, bytes); // cudaMalloc
    cudaMalloc((void **) &b_flat_device, bytes);
    cudaMemcpy(a_flat_device, a_flat, bytes, cudaMemcpyHostToDevice); // cudaMemcpy
    cudaMemcpy(b_flat_device, b_flat, bytes, cudaMemcpyHostToDevice);
    cudai_add<<<N_BLOCKS, N_THREADS>>>(a_flat_device, b_flat_device, cells); // CUDA Kernel
    cudaDeviceSynchronize();                                         
    cudaMemcpy(a_flat, a_flat_device, bytes, cudaMemcpyDeviceToHost); // Copy from device to host
    load_values(a_flat);     // Load matrix with array
    cudaFree(a_flat_device); // FREE
    cudaFree(b_flat_device);
    free(a_flat);
    free(b_flat);
}

void Matrix::cuda_matsub(const Matrix &B) {
    // DECLARE
    size_t bytes, N_BLOCKS, N_THREADS;
    double *a_flat, *b_flat, *a_flat_device, *b_flat_device;

    // INITIALIZE
    a_flat = c_flatten();
    b_flat = B.c_flatten();
    bytes = cells * sizeof(double);
    N_THREADS = 1024;
    N_BLOCKS = cells/1024 + 1;

    cudaMalloc((void **) &a_flat_device, bytes); // cudaMalloc
    cudaMalloc((void **) &b_flat_device, bytes);
    cudaMemcpy(a_flat_device, a_flat, bytes, cudaMemcpyHostToDevice); // cudaMemcpy
    cudaMemcpy(b_flat_device, b_flat, bytes, cudaMemcpyHostToDevice);
    cudai_sub<<<N_BLOCKS, N_THREADS>>>(a_flat_device, b_flat_device, cells); // CUDA Kernel
    cudaDeviceSynchronize();                                         
    cudaMemcpy(a_flat, a_flat_device, bytes, cudaMemcpyDeviceToHost); // Copy from device to host
    load_values(a_flat);     // Load matrix with array
    cudaFree(a_flat_device); // FREE
    cudaFree(b_flat_device);
    free(a_flat);
    free(b_flat);
}

void Matrix::cuda_hadamard(const Matrix &B) {
    size_t bytes, N_BLOCKS, N_THREADS;
    double *a_flat, *b_flat, *a_flat_device, *b_flat_device;

    // INITIALIZE
    a_flat = c_flatten();
    b_flat = B.c_flatten();
    bytes = cells * sizeof(double);
    N_THREADS = 1024;
    N_BLOCKS = cells/1024 + 1;

    cudaMalloc((void **) &a_flat_device, bytes); // cudaMalloc
    cudaMalloc((void **) &b_flat_device, bytes);
    cudaMemcpy(a_flat_device, a_flat, bytes, cudaMemcpyHostToDevice); // cudaMemcpy
    cudaMemcpy(b_flat_device, b_flat, bytes, cudaMemcpyHostToDevice);
    cudai_hadamard<<<N_BLOCKS, N_THREADS>>>(a_flat_device, b_flat_device, cells); // CUDA Kernel
    cudaDeviceSynchronize();                                         
    cudaMemcpy(a_flat, a_flat_device, bytes, cudaMemcpyDeviceToHost); // Copy from device to host
    load_values(a_flat);     // Load matrix with array
    cudaFree(a_flat_device); // FREE
    cudaFree(b_flat_device);
    free(a_flat);
    free(b_flat);
}