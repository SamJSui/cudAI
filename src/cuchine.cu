#include "cuchine.cuh"

void read_file(std::ifstream &ifs, char *filename) {
    ifs.open(filename);
    if (ifs.is_open()) { return; }
    else {
        std::cerr << "ERROR: File did not open!\n";
        exit(EXIT_FAILURE);
    }
}