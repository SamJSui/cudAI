CC = nvcc
INCLUDES = -I include/
CFLAGS = $(INCLUDES)
INPUT = data/input.txt
EXECUTABLES = bin/main
SRC = src/main.cu src/cuchine.cu

all: $(EXECUTABLES)

bin/main: src/*.cu
	$(CC) $(CFLAGS) -o $@ $^

run: bin/main
	$^ $(INPUT)