#include "cuchine.cuh"

Matrix::Matrix() {
    dim = 0;
    shape = (int *) malloc(2 * sizeof(int)); // Rows x Cols
}

Matrix::~Matrix() {
    free(shape);
}

void Matrix::read_file(char *filename) {
    // DECLARE
    std::string line, num;
    int i;

    // FILE HANDLING
    std::ifstream ifs(filename, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: File did not open!\n";
        exit(EXIT_FAILURE);
    }

    // INITIALIZE
    getline(ifs, line); // Get dimensions
    read_dim(line);
    std::istringstream iss(line);

    // READ
    int dim_idx;
    dim_idx = 0;
    while (getline(iss, num, 'x')) {
        try { shape[dim_idx] = stod(num); }
        catch (...) { std::cerr << "ERROR: Input File has invalid values (check the characters)\n"; exit(1); }
        if (dim_idx == dim) break;
        ++dim_idx;
    }
    for (i = 0; i < get_rows(); i++) {
        matrix.push_back(std::vector<double>(get_cols(), 0));
    }
    ifs.close();
}

void Matrix::read_dim(std::string &line) {
    int i;
    for (i = 0; i < line.length(); i++) {
        if (line[i] == 'x') dim++;
        else if (line[i] == ',') line[i] = ' ';
        if (dim > 1) {
            std::cerr << "ERROR: Input File has too many dimensions! (What're trying to do? How do I support n-dimensional data in a .txt)\n"; exit(1);
        }
    }
}

std::vector<double> Matrix::operator[](int idx) {
    if (matrix.size() > 0 && idx > 0 && idx < get_rows()) return matrix[idx];
    else { 
        std::cerr << "ERROR: Invalid index accessing matrix!\n";
        return std::vector<double>();
    }
}