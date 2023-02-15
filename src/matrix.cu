#include "matrix.cuh"

//// PUBLIC

Matrix::Matrix() {
    dim = 0;
    matrix = NULL;
    shape = (int *) malloc(2 * sizeof(int)); // Rows x Cols
}

Matrix::~Matrix() {
    int i;
    dim = 0;
    if (matrix != NULL) {
        for (i = 0; i < get_rows(); i++) { free(matrix[i]); }
    }
    free(shape);
    free(matrix);
}

void Matrix::read_file(char *filename) {
    // READING IN FILE AGAIN
    if (matrix != NULL) {
        int i;
        dim = 0;
        for (i = 0; i < get_rows(); i++) { free(matrix[i]); } // FREE PREVIOUSLY-ALLOCATED
        free(matrix);
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
        try { shape[dim_idx] = stoi(num); } // Reads in rows and columns
        catch (std::exception &e) { 
            fprintf(stderr, "ERROR: Input File has invalid values (check the characters)\n %s\n", e.what()); 
            exit(1); 
        }
        if (dim_idx == dim) break;
        ++dim_idx;
    }

    // MATRIX VALUES
    matrix = (double **) malloc(get_rows() * sizeof(double *));
    fill_values(ifs);

    // CLOSE FILE
    ifs.close();
}

double* Matrix::operator[](int idx) {
    if (get_rows() > 0 && idx > -1 && idx < get_rows()) return matrix[idx];
    else { 
        std::cerr << "ERROR: Invalid index accessing matrix!\n";
        return {};
    }
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

void Matrix::fill_values(std::ifstream &ifs) {
    // DECLARE
    std::string line, num;
    int rows, cols, lineNum, i, j, comma;
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
        for (comma = 0; comma < line.length(); comma++) { if (line[comma] == ',') line[comma] = ' '; }
        iss = std::istringstream(line);
        matrix[i] = (double *) malloc(cols * sizeof(double));
        while (getline(iss, num, ' ')) {
            if (j == cols) { std::cerr << "ERROR: Line " << lineNum << " has too many values has too many values in the row!\n"; exit(1); }
            try { matrix[i][j] = stod(num); }
            catch (std::exception &e) {
                fprintf(stderr, "ERROR: Line %d has input data has invalid characters within the values!\n %s\n", lineNum, e.what());
                exit(1);
            }
            j++;
        }
        j = 0;
        i++;
        lineNum++;
    }
}