# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -I/usr/include/boost -L/usr/include/boost -fopenmp
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 
BIN = gcc
.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : main

main: main.cpp FOV_models.cpp 
	g++ -o main main.cpp $(OPT)




# TODO: add targets for building executables

.PHONY: clean
clean:
	rm -f main
