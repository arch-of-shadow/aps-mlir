# Makefile for DECA C Reference Implementation

CC = gcc
CFLAGS = -Wall -Wextra -std=c99
OPTFLAGS = -O2
DEBUGFLAGS = -g -O0

# Targets
all: test_c_reference

test_c_reference: test_c_reference.c deca_decompress_simple.c
	$(CC) $(CFLAGS) $(OPTFLAGS) test_c_reference.c -o test_c_reference

debug: test_c_reference.c deca_decompress_simple.c
	$(CC) $(CFLAGS) $(DEBUGFLAGS) test_c_reference.c -o test_c_reference_debug

# Run tests
test: test_c_reference
	./test_c_reference

# Generate assembly
asm: deca_decompress_simple.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -S -fverbose-asm deca_decompress_simple.c -o deca_decompress_simple.s

# Clean
clean:
	rm -f test_c_reference test_c_reference_debug *.s *.o

.PHONY: all test clean asm debug
