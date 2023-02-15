# CUchine DArning

Library of parallelized mathematical and machine learning operations


## About 

**Origin**: *I came up with the name in the middle of an automata lecture*

**Purpose**: *I felt bad whenever I used SymboLab (online math solver), but I also did not want to take out pen(cil) and paper*

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