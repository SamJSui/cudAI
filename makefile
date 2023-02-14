CC = nvcc
INCLUDES = -I include/
CFLAGS = $(INCLUDES)
INPUT = data/*.txt
EXECUTABLES = bin/main

all: $(EXECUTABLES)

bin/main: src/*.cu
	$(CC) $(CFLAGS) -o $@ $^

run: bin/main
	$^ $(INPUT)

val: bin/main
	valgrind $^ $(INPUT)
