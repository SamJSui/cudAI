CC = nvcc
INCLUDES = -I include/
CFLAGS = $(INCLUDES) -g
INPUT = data/1x3.txt data/1x3_1.txt
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
