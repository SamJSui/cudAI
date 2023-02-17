#include "cudai.cuh"

int main(int argc, char **argv) {
    // ERROR CHECK
    if (argc < 2) { std::cerr << "ERROR: Missing input file\n"; }

    // DECLARE
    char *filename1, *filename2;

    // INITIALIZE
    filename1 = argv[1];
    filename2 = argv[2];

    // MATRIX
    Matrix A, B, C;
    A.read_file(filename1);
    B.read_file(filename2);

    std::vector<double> t({10, 11});
    A.hstack(B);
    A.hstack(t);
    std::cout << A;
    return 0;
}