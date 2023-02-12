#include "cuchine.cuh"

int main(int argc, char **argv) {
    // ERROR CHECK / FILE HANDLING
    if (argc < 2) { std::cerr << "ERROR: Missing input file\n"; }

    // DECLARE
    char *filename;
    std::ifstream ifs;

    // INITIALIZE
    filename = argv[1];
    read_file(ifs, filename);

    

    return 0;
}