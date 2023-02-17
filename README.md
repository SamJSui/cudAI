# cudAI

## About 

**cudAI** is a CUDA C/C++ that parallelizes linear algebra and machine learning algorithms to provide efficient computing. The goal is to combine aspects of Python's NumPy and Scikit-Learn into a single library, providing matrix operations and model training in a C/C++ environment.

### Built with

- CUDA C/C++

## Getting Started



### Installation

1. `git clone https://github.com/SamJSui/CUchine_DArning`
2. `cd CUchine_DArning`
3. Modify the following (if needed):
   1. **makefile**, if making changes to command-line execution
   2. **src/main.cu**, to use the library
4. If making a new file, copy `include/` over and add it to execution
   1. `nvcc -I include/ -o new_main new_main.cu `

## Usage

[Documentation](https://github.com/SamJSui/CUchine_DArning) (UNDER CONSTRUCTION)

## Roadmap

- [ ] Documentation
- [ ] Matrix Operations
  - [ ] Matrix Addition/Subtraction
  - [ ] Matrix Multiplication
  - [ ] Matrix Transposition
- [ ] Gaussian Elimination
- [ ] Linear Regression (class Model)