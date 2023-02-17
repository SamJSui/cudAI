#include "matrix.cuh"

//// PUBLIC

Matrix::Matrix() {
    dim = 0;
    device = 0;                                    // Device check
    cudaGetDeviceCount((int *) &device);
}

Matrix::~Matrix() {
    dim = 0;
}

void Matrix::read_file(char *filename, char delimiter) {
    // READING IN FILE AGAIN
    if (matrix.size() > 0) {
        int i;
        dim = 0;
        for (i = 0; i < get_rows(); i++) { matrix[i].clear(); } // CLEAR
    }

    // DECLARE
    std::string line, num;
    int dim_idx;

    // FILE HANDLING
    std::ifstream ifs(filename, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: File did not open!\n";
        exit(EXIT_FAILURE);
    }

    // INITIALIZE
    getline(ifs, line); // Get dimensions
    read_dim(line); // Clean ',' and increment number of dimensions
    std::istringstream iss(line);
    dim_idx = 0;

    // MATRIX SHAPE
    while (getline(iss, num, 'x')) {
        try { shape.push_back(stoi(num)); } // Reads in rows and columns
        catch (std::exception &e) { 
            fprintf(stderr, "ERROR: Input File has invalid values (check the characters)\n %s\n", e.what()); 
            exit(1); 
        }
        if (dim_idx == dim) break;
        ++dim_idx;
    }

    // MATRIX VALUES
    fill_values(ifs, delimiter);

    // CLOSE FILE
    ifs.close();
}

std::vector<double> Matrix::operator[](long long idx) const {
    if (get_rows() > 0 && idx > -1 && idx < get_rows()) return matrix[idx];
    else { 
        std::cerr << "ERROR: Invalid index accessing matrix!\n";
        return {};
    }
}

std::ostream& operator<< (std::ostream &os, const Matrix &M) {
    int i, j;
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

void Matrix::operator+= (const Matrix &B) {
    // ERROR CHECK
    if (get_rows() != B.get_rows() || get_cols() != B.get_cols()) {
        fprintf(stderr, "ERROR: Invalid '+=' operation!\n");
        exit(1);
    }
    if (device) { cuda_matadd(B); }
    else matadd(B);
}

void Matrix::operator-= (const Matrix &B) {
    // ERROR CHECK
    if (get_rows() != B.get_rows() || get_cols() != B.get_cols()) {
        fprintf(stderr, "ERROR: Invalid '+=' operation!\n");
        exit(1);
    }

    if (device) cuda_matsub(B);
    else matsub(B);
}

double* Matrix::c_flatten() const {
    // DECLARE
    int rows, cols, i, j, idx;
    double *tmp;

    // INITIALIZE
    rows = get_rows();
    cols = get_cols();
    tmp = (double *) malloc(rows * cols * sizeof(double));

    // LOAD
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = rows * i + j;
            tmp[idx] = matrix[i][j];
        }
    }
    return tmp;
}

//// PRIVATE

// Reads numbers of 'x'
// TO DO: N-DIMENSIONS IS NOT SUPPORTED (only 2D)
void Matrix::read_dim(std::string &line) {
    int i;
    for (i = 0; i < line.length(); i++) {
        if (line[i] == 'x') dim++;
        else if (line[i] == ',') line[i] = ' ';
        if (dim > 1) { 
            fprintf(stderr, "ERROR: Input File has too many dimensions!\n");
            exit(1);
        }
    }
}

void Matrix::fill_values(std::ifstream &ifs, char delimiter) {
    // DECLARE
    std::string line, num;
    int rows, cols, lineNum, i, j;
    std::istringstream iss;

    // INITIALIZE
    rows = get_rows(), cols = get_cols();
    i = 0, j = 0;
    lineNum = 2;

    while (getline(ifs, line)) {
        if (i == rows) { 
            fprintf(stderr, "ERROR: Input file has too many rows!\n");
            exit(1);
        }
        iss = std::istringstream(line);
        std::vector<double> tmp_row;
        while (getline(iss, num, delimiter)) {
            if (j == cols) { std::cerr << "ERROR: Line " << lineNum << " has too many values has too many values in the row!\n"; exit(1); }
            try { tmp_row.push_back(stod(num)); }
            catch (std::exception &e) {
                fprintf(stderr, "ERROR: Line %d has input data has invalid characters within the values!\n %s\n", lineNum, e.what());
                exit(1);
            }
            j++;
        }
        matrix.push_back(tmp_row);
        if (j != cols) {
            fprintf(stderr, "ERROR: Not enough values in input file!\n");
            exit(1);
        }
        j = 0;
        i++;
        lineNum++;
    }
    if (i != rows) {
        fprintf(stderr, "ERROR: Not enough values in input file!\n");
        exit(1);
    }
}

void Matrix::load_values(double *d) {
    int i, j, rows;
    rows = get_rows();
    for (i = 0; i < rows; i++) {
        for (j = 0; j < get_cols(); j++) {
            int idx = rows * i + j;
            matrix[i][j] = d[idx];
        }
    }
}

void Matrix::copy(const Matrix &B){
    int i, j;
    shape.resize(2);
    shape[0] = B.get_rows();
    shape[1] = B.get_cols();
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

// CUDA
void Matrix::cuda_matadd(const Matrix &B) {
    // DECLARE
    int rows, cols, bytes;
    double *a_flat, *b_flat, *a_flat_device, *b_flat_device;

    // INITIALIZE
    rows = get_rows();
    cols = get_cols();
    a_flat = c_flatten();
    b_flat = B.c_flatten();
    bytes = rows * cols * sizeof(double);

    cudaMalloc((void **) &a_flat_device, bytes); // cudaMalloc
    cudaMalloc((void **) &b_flat_device, bytes);
    cudaMemcpy(a_flat_device, a_flat, bytes, cudaMemcpyHostToDevice); // cudaMemcpy
    cudaMemcpy(b_flat_device, b_flat, bytes, cudaMemcpyHostToDevice);
    cuchine_matadd<<<1, 4>>>(a_flat_device, b_flat_device, rows*cols); // CUDA Kernel
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
    int rows, cols, bytes;
    double *a_flat, *b_flat, *a_flat_device, *b_flat_device;

    // INITIALIZE
    rows = get_rows();
    cols = get_cols();
    a_flat = c_flatten();
    b_flat = B.c_flatten();
    bytes = rows * cols * sizeof(double);

    cudaMalloc((void **) &a_flat_device, bytes); // cudaMalloc
    cudaMalloc((void **) &b_flat_device, bytes);
    cudaMemcpy(a_flat_device, a_flat, bytes, cudaMemcpyHostToDevice); // cudaMemcpy
    cudaMemcpy(b_flat_device, b_flat, bytes, cudaMemcpyHostToDevice);
    cuchine_matsub<<<1, 4>>>(a_flat_device, b_flat_device, rows*cols); // CUDA Kernel
    cudaDeviceSynchronize();                                         
    cudaMemcpy(a_flat, a_flat_device, bytes, cudaMemcpyDeviceToHost); // Copy from device to host
    load_values(a_flat);     // Load matrix with array
    cudaFree(a_flat_device); // FREE
    cudaFree(b_flat_device);
    free(a_flat);
    free(b_flat);
}