class Matrix {
    public:
        void read_input(std::ifstream &ifs);
        void get_rows();
        void get_cols();
        void get_shape();
    private:
        int rows, cols;
};