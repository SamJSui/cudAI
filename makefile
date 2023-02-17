CC = nvcc
INCLUDES = -I include/
CFLAGS = $(INCLUDES) -g
INPUT = data/2x2.txt data/2x2_1.txt
EXECUTABLES = bin/main

all: $(EXECUTABLES)

bin/main: src/*.cu
	$(CC) $(CFLAGS) -o $@ $^ include/*.cu

run:
	$(CC) $(CFLAGS) -o $(EXECUTABLES) src/main.cu include/*.cu
	$(EXECUTABLES) $(INPUT)

val: bin/main
	valgrind $^ $(INPUT)

gdb: bin/main
	gdb $^ $(INPUT)
